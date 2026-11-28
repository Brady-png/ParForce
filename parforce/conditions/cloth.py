"""First-frame cloth projection and local-grip XPBD controls."""
from pathlib import Path
import json

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .firstview_v2 import digest
from .pipeline import video
from .raster import raster_accelerated
from .rigid_particles import fixed_marker_image
from .cloth_solver import constrain, edges_and_bends, rollout as wind_rollout
from .material_render import farthest


def refine_grounded_contour(image, outline, hanging=False):
    """Tighten the model's coarse cloth outline against first-frame image evidence."""
    seed = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(seed, [np.asarray(outline, dtype=np.int32)], 1)
    x, y, width, height = cv2.boundingRect(seed)
    if min(width, height) < 12:
        raise ValueError('Coarse cloth outline is too small to refine')
    margin = max(12, round(.29 * max(width, height)))
    distance = cv2.distanceTransform(seed, cv2.DIST_L2, 5)
    core = distance > .7 * distance.max()
    compatible = np.ones_like(core)
    if hanging:
        chroma = image[:,:,2].astype(np.int16)-image[:,:,0]
        signed = chroma*np.sign(np.median(chroma[core]))
        lower = np.quantile(signed[core],.25)
        if lower>12:
            compatible = signed >= lower*.4
            core &= compatible

    def segment(padding):
        top, left = max(0, y-padding), max(0, x-padding)
        bottom, right = min(y+height+padding, seed.shape[0]), min(x+width+padding, seed.shape[1])
        labels = np.full_like(seed, cv2.GC_BGD)
        labels[top:bottom, left:right] = cv2.GC_PR_BGD
        labels[seed != 0] = cv2.GC_PR_FGD
        labels[~compatible] = cv2.GC_BGD
        labels[core] = cv2.GC_FGD
        cv2.setRNGSeed(91)
        try:
            cv2.grabCut(image, labels, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
        except cv2.error as error:
            raise ValueError('First-frame segmentation could not fit the proposed garment') from error
        foreground = np.uint8((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD))
        _, components = cv2.connectedComponents(foreground)
        component = np.bincount(components[core]).argmax()
        mask = np.uint8(components == component)
        clipped = ((top > 0 and mask[top].any()) or (bottom < seed.shape[0] and mask[bottom-1].any())
                   or (left > 0 and mask[:, left].any()) or (right < seed.shape[1] and mask[:, right-1].any()))
        return mask, component, clipped

    mask, component, clipped = segment(margin)
    if component and clipped:
        mask, component, _ = segment(2 * margin)
    if component == 0 or not .52 < mask.sum() / seed.sum() < 2.75:
        raise ValueError('First-frame segmentation did not resolve the targeted cloth')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(contour, 1.3, True)[:, 0]
    return polygon.tolist(), contour[:, 0]


def fitted_sheet(contour, hanging=False):
    """Triangulate a photographed silhouette on an estimated support surface."""
    h,w = 256,384
    mask = np.zeros((h,w),np.uint8)
    cv2.fillPoly(mask,[np.array(contour,dtype=np.int32)],255)
    stride = 3 if np.count_nonzero(mask) < 5000 else 4
    x,y = np.meshgrid(np.arange(0,w,stride),np.arange(0,h,stride))
    uv = np.stack((x,y),-1).reshape(-1,2)
    mask_index = (mask[uv[:,1],uv[:,0]]!=0).reshape(x.shape)
    ij = np.arange(x.size).reshape(x.shape)
    a,b,c,d = ij[:-1,:-1],ij[:-1,1:],ij[1:,:-1],ij[1:,1:]
    raw = np.concatenate([np.stack(t,axis=-1).reshape(-1,3) for t in [(a,b,d),(a,d,c)]])
    raw = raw[np.all(mask_index.ravel()[raw],axis=1)]
    ids,faces = np.unique(raw,return_inverse=True)
    faces = faces.reshape(-1,3).astype(np.int32)
    uv = uv[ids]
    K = np.array([[448.,0.,191.5],[0.,448.,127.5],[0.,0.,1.]])
    if hanging:
        # Rest geometry approximates the observed loaded hanging shape. Keep
        # each vertex on its original camera ray, including assumed pleats.
        xi = (uv[:,0]-uv[:,0].min())/np.ptp(uv[:,0])
        depth = .81 + .015*np.cos(6*np.pi*xi)
        camera = (np.c_[uv+.5,np.ones(len(uv))]@np.linalg.inv(K).T)*depth[:,None]
        rotation = np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
        lift = np.array([0.,0.,.8])
        return camera@rotation+lift, faces, uv, mask, K, rotation, -lift@rotation.T
    angle = np.deg2rad(64)
    s,c = np.sin(angle),np.cos(angle)
    rotation = np.array([[1.,0.,0.],[0.,-s,-c],[0.,c,-s]])
    offset = np.array([0.,0.,.81])
    camera_origin = -offset@rotation
    rays = np.c_[uv+.5,np.ones(len(uv))]@np.linalg.inv(K).T@rotation
    xyz = camera_origin + ((.002-camera_origin[2])/rays[:,2])[:,None]*rays
    return xyz, faces, uv, mask, K, rotation, offset


def control_paths(xyz, uv, action, fixed_xy=()):
    """Bind the commanded grip and anchor patches once on the fitted sheet."""
    span = np.ptp(xyz, axis=0)

    def nearest_patch(xy, count):
        distance = np.linalg.norm(uv - np.asarray(xy), axis=1)
        ids = np.argsort(distance)[:count]
        if len(ids) < count or distance[ids].max() > 19:
            raise ValueError("Cloth grip or anchor is outside the fitted material surface")
        return ids

    fixed = np.unique(np.concatenate([nearest_patch(xy, 5) for xy in fixed_xy])) if fixed_xy else np.empty(0, dtype=int)
    if any(item.get("fold") for item in action):
        box = cv2.boxPoints(cv2.minAreaRect(np.asarray(xyz[:, :2], dtype=np.float32)))
        normals = (box[1:3] - box[:2]).astype(float)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    groups = []
    for item in action:
        ids = nearest_patch(item["xy"], item.get("vertices", 7))
        delta = np.asarray(item["translation_fraction"]) * span[0]
        arc = item.get("lift_arc", 0.) * span[0]
        fold = item.get("fold")
        if fold is not None:
            axis = int(fold["axis"])
            normal = normals[np.argmax(np.abs(normals[:, axis]))]
            normal = normal * np.sign(normal[axis])
            coordinates = xyz[:, :2] @ normal
            width = float(np.ptp(coordinates))
            edge = float(np.mean(coordinates[ids]))
            center = float(coordinates.min() + fold["hinge_fraction"] * width)
            sign = 1 if edge > center else -1
            if abs(edge - center) < .07 * width:
                raise ValueError("The moving edge must be separated from its fold axis")
            if fold.get("hem_grasp", False):
                tangent = xyz[:, :2] @ np.array([-normal[1], normal[0]])
                band = (np.abs(coordinates - edge) < .045 * width)
                band &= np.abs(tangent - tangent[ids].mean()) < .18 * np.ptp(tangent)
                if band.any():
                    candidates = np.flatnonzero(band)
                    sample = np.argsort(tangent[candidates])
                    positions = np.linspace(0, len(sample) - 1, min(len(sample), 18), dtype=int)
                    ids = np.union1d(ids, candidates[sample[positions]])
            fold = (normal, sign, center, float(fold["angle_rad"]), float(fold["clearance_m"]))
        groups.append((ids, delta, item["start"], item["ramp"], arc, item.get("return_start"), item.get("release_at"), fold))
    all_ids = np.concatenate([group[0] for group in groups])
    if len(np.unique(all_ids)) != len(all_ids) or np.intersect1d(fixed, all_ids).size:
        raise ValueError("Moving grips and fixed anchor patches must be distinct")
    return groups, fixed


def grip_targets(base, delta, ease, arc, fold=None):
    """Move local control points, or rotate them together around a fold axis."""
    ease = np.asarray(ease)
    if fold is None:
        targets = base + ease[..., None, None] * delta
        targets[..., 2] += arc * np.sin(np.pi * ease[..., None])
        return targets
    normal, sign, hinge, angle, clearance = fold
    theta = ease[..., None] * angle
    c, s = np.cos(theta), sign * np.sin(theta)
    d = base[:, :2] @ normal - hinge
    z = base[:, 2] - .002
    targets = np.broadcast_to(base, ease.shape + base.shape).copy()
    normal_shift = d*(c - 1.) - z*s
    targets[..., :2] += normal_shift[..., None] * normal
    targets[..., 2] = .002 + d*s + z*c + clearance * ease[..., None]
    return targets


def rollout(xyz, faces, controls, fixed=()):
    """Drive local patches along the requested phases while the rest of the sheet deforms."""
    (edges, lengths), (bends, bend_lengths) = edges_and_bends(faces, xyz)
    forbidden = {tuple(e) for e in np.concatenate((edges, bends))}
    pos, velocity = xyz.copy(), np.zeros_like(xyz)
    result = [xyz.copy()]
    dt = 1 / 144
    onset = min(group[2] for group in controls)
    for step in range(1, 577):
        time = step * dt
        if time > onset:
            previous = pos.copy()
            inv = np.ones(len(pos))
            pgoals = []
            if len(fixed):
                inv[fixed] = 0
                pgoals.append((fixed, xyz[fixed]))
            for ids, delta, start, ramp, arc, return_start, release_at, fold in controls:
                if time <= start or (release_at is not None and time > release_at):
                    continue
                progress = np.clip((time - start) / ramp, 0, 1)
                if return_start is not None and time > return_start:
                    progress = 1 - np.clip((time - return_start) / ramp, 0, 1)
                ease = progress * progress * (3 - 2 * progress)
                targets = grip_targets(xyz[ids], delta, ease, arc, fold)
                pgoals.append((ids, targets))
                inv[ids] = 0
            pos += inv[:, None] * (.97 * velocity + dt * np.array([0., 0., -9.81])) * dt
            edge_lam = np.zeros(len(edges))
            bend_lam = np.zeros(len(bends))
            for iteration in range(10):
                for ids, targets in pgoals:
                    pos[ids] = targets
                constrain(pos, inv, edges, lengths, edge_lam, .00050)
                constrain(pos, inv, bends, bend_lengths, bend_lam, .20)
                if iteration in (3, 7):
                    pairs = cKDTree(pos).query_pairs(.0028, output_type="ndarray")
                    pairs = np.asarray([p for p in pairs if tuple(p) not in forbidden], dtype=np.int32).reshape(-1, 2)
                    rest = np.full(len(pairs), .0028)
                    constrain(pos, inv, pairs, rest, np.zeros(len(pairs)), .0007, True)
                pos[:, 2] = np.maximum(pos[:, 2], .002)
            velocity = (pos - previous) / dt
            velocity[pos[:, 2] <= .0021, :2] *= .48
        if step % 12 == 0:
            result.append(pos.copy())
    return np.asarray(result), edges, lengths


def sample_material(initial, faces, requested):
    """Bind near-uniform reference samples to material faces for the whole clip."""
    triangles = initial[faces]
    area = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
    rng = np.random.default_rng(91)
    choices = rng.choice(len(faces),requested*8,p=area/area.sum())
    u,v = np.sqrt(rng.random(len(choices))),rng.random(len(choices))
    weights = np.c_[1-u,u*(1-v),u*v]
    candidates = np.einsum('ni,nik->nk',weights,triangles[choices])
    selected = cKDTree(candidates).query(farthest(candidates,requested))[1]
    return choices[selected],weights[selected]


def render_conditions(camera_vertices, faces, K, image, output, requested):
    """Render PSS from the fixed material samples and SDS from surface faces."""
    face_ids,barycentric = sample_material(camera_vertices[0],faces,requested)
    positions = np.einsum('ni,tnik->tnk',barycentric,camera_vertices[:,faces[face_ids]])
    shape = image.shape[:2]
    near,far = np.quantile(positions[...,2],[.01,.99])
    particles = np.asarray([fixed_marker_image(points,K,shape,near,far) for points in positions])
    depths = np.asarray([raster_accelerated(points,faces,K,shape) for points in camera_vertices])
    normalized = np.where(depths>0,np.clip(.33+.54*(far-depths)/max(far-near,.003),.18,.93),0)
    rgb_depth = np.repeat(np.uint8(np.rint(normalized*255))[...,None],3,axis=-1)
    video(output/'particles.mp4',particles)
    video(output/'depth.mp4',rgb_depth)
    video(output/'overview.mp4',np.asarray([np.hstack((image,p,d)) for p,d in zip(particles,rgb_depth,strict=True)]))
    cv2.imwrite(str(output/'first_frame.png'),image)
    return positions,depths,face_ids,barycentric


def run(row, output_root):
    cv2.setNumThreads(1)
    out = Path(output_root)/row['case']
    out.mkdir(parents=True,exist_ok=True)
    image = cv2.imread(row['image'])
    if image is None or image.shape[:2]!=(256,384):
        raise ValueError('Cloth landmarks require a 384x256 source frame')
    hanging = row.get('material')=='cloth_wind'
    xyz,faces,pixels,mask,K,rotation,offset = fitted_sheet(row['contour'],hanging)
    fixed = np.empty(0,dtype=np.int32)
    if hanging:
        fixed = np.flatnonzero(pixels[:,1]<pixels[:,1].min()+.075*np.ptp(pixels[:,1]))
        if len(fixed)<3:
            raise ValueError('Hanging sheet must have a resolved attachment edge')
        motion,edges,lengths = wind_rollout(xyz,faces,np.empty(0,dtype=int),fixed,
            np.zeros(3),True,wind=np.asarray(row['wind_world']),gravity_z=0,hold_until_grip=True,
            loaded_rest_rate=1.5,gust_end=2.4)
    else:
        controls, fixed = control_paths(xyz, pixels, row['grips'], row.get('fixed_xy', ()))
        motion, edges, lengths = rollout(xyz, faces, controls, fixed)
    camera_vertices = motion@rotation.T+offset
    # Choose density by the projected material area with a moderate count ceiling.
    requested = row.get('points',int(np.clip(round((mask>0).sum()/16),420,2200)))
    positions,depths,choices,bary = render_conditions(camera_vertices,faces,K,image,out,requested)
    np.savez_compressed(out/'states.npz',positions=positions,vertices=motion,faces=faces,
                        face_ids=choices,barycentric=bary,depth_z=depths,K=K,rotation=rotation,
                        offset=offset,times_s=np.arange(49)/12,fixed=fixed)
    fit = float(((depths[0]>0)&(mask>0)).sum()/((depths[0]>0)|(mask>0)).sum())
    ratios = np.linalg.norm(motion[:,edges[:,0]]-motion[:,edges[:,1]],axis=-1)/lengths
    stretch = float(np.quantile(ratios,.99))
    motion_scale = float(np.quantile(np.linalg.norm(motion-xyz,axis=-1),.95))
    checks = {'Initial material silhouette IoU is at most 0.90':bool(fit>.9),
              'Excessive edge stretch':bool(stretch<1.4),
              'No measurable response to wind or grasps':bool(motion_scale>.008),
              'Non-finite material state':bool(np.isfinite(motion).all())}
    if len(fixed):
        checks['Attachment drifted'] = bool(np.max(np.abs(motion[:,fixed]-xyz[fixed]))<1e-8)
    if not hanging:
        checks['Cloth passed below its support'] = bool(motion[...,2].min()>=.002-1e-8)
    report = dict(case=row['case'],prompt=row['prompt'],image_sha256=digest(row['image']),
                  backend='anchored wind XPBD' if hanging else 'local-grip XPBD',
                  support='assumed loaded sheet with fixed top edge' if hanging else 'assumed planar support through fitted silhouette',
                  fit_iou=fit,stretch_p99=stretch,frames=len(positions),fps=12,
                  particle_count=requested,verified_numeric=all(checks.values()),checks=checks,
                  p95_motion_m=motion_scale,
                  visual_pass=None)
    (out/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
