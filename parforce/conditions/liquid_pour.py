"""First-frame-only hollow-vessel reconstruction and position-based liquid rollout."""
from pathlib import Path
import json

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .firstview_v2 import digest
from .pipeline import video
from .raster import raster_accelerated
from .material_render import farthest, mesh_surface_candidates, raster_subpixel_points


def receiver_surface(radius, height, floor, width_profile, n=96):
    """Build matching outer and inner walls around the open rim."""
    angles = np.arange(n)*2*np.pi/n
    sections = np.linspace(0,1,9)
    inner_sections = sections[::-1]
    levels = np.r_[sections*height,floor+inner_sections*(height-floor)]
    radii = np.r_[radius*width_profile(sections),radius*width_profile(levels[9:]/height)-.003]
    xyz = np.stack((radii[:,None]*np.cos(angles),np.broadcast_to(levels[:,None],(len(levels),n)),
                    radii[:,None]*np.sin(angles)),axis=-1).reshape(-1,3)
    centers = np.array([[0,0,0],[0,floor,0]])
    rings = np.arange(len(xyz)).reshape(-1,n)
    a,b = rings[:-1].ravel(),np.roll(rings[:-1],-1,axis=1).ravel()
    c,d = np.roll(rings[1:],-1,axis=1).ravel(),rings[1:].ravel()
    edges = np.r_[np.c_[a,b,c],np.c_[a,c,d]]
    floor_faces = np.r_[np.c_[rings[0],np.full(n,len(xyz)),np.roll(rings[0],-1)],
                        np.c_[rings[-1],np.roll(rings[-1],-1),np.full(n,len(xyz)+1)]]
    return np.r_[xyz,centers],np.r_[edges,floor_faces],rings[8],rings[0]


def fit_vessel(rim, base, middle_width=None):
    """Fit one upright tapered vessel under a single pinhole camera.

    Landmarks are left, top, right, bottom in output pixels for the rim
    and visible lower contour. A nominal rim radius fixes scale.
    """
    rim, base = np.asarray(rim, float), np.asarray(base, float)
    camera = np.array([[340., 0., 191.5], [0., 340., 127.5], [0., 0., 1.]])
    radius = .045
    floor = .005
    taper = float(np.clip((base[2] - base[0]) / (rim[2] - rim[0]), .48, 1.05))
    middle = (1+taper)/2 if middle_width is None else np.clip(middle_width/(rim[2]-rim[0]),.55,1.24)

    def width_profile(level):
        t = np.clip(level,0,1)
        return taper+(1-taper)*t+4*(middle-(1+taper)/2)*t*(1-t)

    def geometry(params):
        height, angle, tx, ty, tz = params
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[1., 0., 0.], [0., -c, -s], [0., -s, c]])
        verts, faces, mouth, foot = receiver_surface(radius,height,floor,width_profile)
        return verts, faces, rotation, np.array([tx, ty, tz]), mouth, foot

    def residual(params):
        verts, _, rotation, offset, mouth, bottom = geometry(params)
        uvz = (verts @ rotation.T + offset) @ camera.T
        uv = uvz[:, :2]/uvz[:, 2:]
        rims = uv[mouth]
        foot = uv[bottom]
        measure = np.r_[rims.min(0), rims.max(0), foot.min(0), foot.max(0)]
        # Match the center and visible lower extent of the foot. Its hidden
        # back is much less reliable in photographs than its front boundary.
        return (measure - np.r_[rim, base]) * np.array([1, 1, 1, 1, 1, .25, 1, 1])

    tz = 2 * camera[0, 0] * radius / (rim[2]-rim[0])
    tx = (.5*(rim[0]+rim[2])-camera[0, 2])*tz/camera[0, 0]
    ty = (base[3]-camera[1, 2])*tz/camera[1, 1]
    start = [.11, .13, tx, ty, tz]
    fit = least_squares(residual, start, bounds=([.035,.003,-1.,-1.,.11],
                                               [.35,.78,1.,1.,2.0]))
    return (*geometry(fit.x)[:4], camera, radius, width_profile, floor, float(fit.x[0]),
            float(np.sqrt(np.mean(residual(fit.x)**2))))


def activate(row, rotation, offset, camera, height, radius):
    """Translate a visible or assumed nozzle pixel into the receiver frame."""
    ray = np.linalg.solve(camera, np.r_[row['nozzle_xy'], 1.])
    local_ray = ray @ rotation
    origin = -offset @ rotation
    nozzle = origin - origin[2]/local_ray[2]*local_ray
    dy = nozzle[1]-height
    if dy <= .005:
        raise ValueError('Nozzle must start above the reconstructed rim plane')
    speed_down = .15
    dt = (-speed_down + np.sqrt(speed_down**2+2*9.81*dy))/9.81
    crossing = nozzle*.20
    if 'stream_rim_xy' in row:
        rim_ray = np.linalg.solve(camera,np.r_[row['stream_rim_xy'],1.])@rotation
        if abs(rim_ray[1])>1e-6:
            intercept = origin+(height-origin[1])/rim_ray[1]*rim_ray
            crossing = intercept
    limit = .69*radius
    crossing *= min(1.,limit/max(np.linalg.norm(crossing[[0,2]]),1e-6))
    velocity = np.array([(crossing[0]-nozzle[0])/dt, -speed_down, (crossing[2]-nozzle[2])/dt])
    return nozzle, velocity


def handle_surface(box, reference_depth, camera):
    """Reproject an observed loop handle into an elliptical surface tube."""
    left,top,right,bottom = np.asarray(box,float)
    center = np.array([(left+right)/2,(top+bottom)/2])
    extent = np.array([(right-left)/2,(bottom-top)/2])
    angle = np.linspace(0,2*np.pi,48,endpoint=False)
    ring = center+np.c_[np.cos(angle),np.sin(angle)]*extent
    along = np.c_[-np.sin(angle)*extent[0],np.cos(angle)*extent[1]]
    outward = np.c_[along[:,1],-along[:,0]]
    outward /= np.linalg.norm(outward,axis=1,keepdims=True)
    cross = np.linspace(0,2*np.pi,8,endpoint=False)
    tube_px = float(np.clip(min(extent)*.17,.9,3.2))
    xy = ring[:,None,:]+outward[:,None,:]*tube_px*np.cos(cross)[None,:,None]
    rays = np.concatenate((xy,np.ones((*xy.shape[:2],1))),axis=-1)@np.linalg.inv(camera).T
    depths = reference_depth+np.sin(cross)*tube_px*reference_depth/camera[0,0]
    vertices = (rays*depths[None,:,None]).reshape(-1,3)
    grid = np.arange(len(vertices)).reshape(48,8)
    a,b = grid.ravel(),np.roll(grid,-1,axis=1).ravel()
    c = np.roll(np.roll(grid,-1,axis=1),-1,axis=0).ravel()
    d = np.roll(grid,-1,axis=0).ravel()
    return vertices,np.r_[np.c_[a,b,c],np.c_[a,c,d]]


def simulate(row, radius, width_profile, floor, height, nozzle, velocity):
    """Advect persistent volume particles with gravity, contact and PBF constraints."""
    inner = radius-.003
    rng = np.random.default_rng(row.get('seed', 42))
    dt = 1/120
    spacing = .0053
    kernel = 2.4*spacing
    mass = 1000*spacing**3
    poly6 = 315/(64*np.pi*kernel**9)
    spiky = -45/(np.pi*kernel**6)
    pad = spacing*.45

    def cavity_radius(y):
        return radius*width_profile(y/height)-.003-pad

    def contacts(candidate, deposited):
        candidate[deposited, 1] = np.maximum(candidate[deposited, 1], floor+pad)
        limit = cavity_radius(candidate[:, 1])
        radial = np.linalg.norm(candidate[:, (0, 2)], axis=1)
        touching = deposited & (radial>limit) & (candidate[:, 1]<height)
        candidate[touching, 0] *= limit[touching]/radial[touching]
        candidate[touching, 2] *= limit[touching]/radial[touching]
        return candidate

    level = row.get('initial_fill', 0.)*height
    if level>floor+spacing:
        grid = np.arange(-inner, inner+spacing, spacing)
        x = np.stack(np.meshgrid(grid, np.arange(floor+pad, level, spacing),
                                 grid, indexing='ij'), axis=-1).reshape(-1,3)
        x = x[np.linalg.norm(x[:, (0,2)], axis=1)<cavity_radius(x[:,1])-.001]
    else:
        x = np.empty((0, 3))
    v = np.zeros_like(x)
    deposited = np.ones(len(x), bool)
    start, stop = row.get('emission_seconds', [.12, 2.46])
    nozzle_radius = row.get('nozzle_radius', .0042)
    flow_count = row.get('particles_per_emission', 14)
    fill_limit = .84 if row.get('liquid_stop_condition') == 'before_rim' else .91
    capacity_heights = np.arange(floor+pad,fill_limit*height,spacing)
    available = np.sum(np.pi*cavity_radius(capacity_heights)**2/spacing**2)-len(x)
    # Unspecified duration is bounded by the receiving volume. Preserve the
    # observed fill and let the solver track actual retained liquid.
    stop = min(stop,start+max(.25,available*dt/max(1.,.5*flow_count)))

    def emit(time, count):
        angles = rng.uniform(0, 2*np.pi, count)
        radii = nozzle_radius*np.sqrt(rng.random(count))
        return nozzle + np.c_[radii*np.cos(angles)+.0012*np.sin(time*3),
                              np.zeros(count), radii*np.sin(angles)]

    if row.get('already_pouring', False):
        # Recover the in-flight portion of a stream visible at time zero.
        # Its age follows the travel time from the observed outlet to the rim.
        arrival = (velocity[1]+np.sqrt(velocity[1]**2+2*9.81*(nozzle[1]-height)))/9.81
        count = max(4, round(.5*flow_count*arrival/dt))
        age = np.linspace(0,arrival,count,endpoint=False)
        fresh = emit(0,count)+age[:,None]*velocity
        fresh[:,1] -= .5*9.81*age**2
        stream_velocity = np.tile(velocity,(count,1))
        stream_velocity[:,1] -= 9.81*age
        x = np.r_[x,fresh]
        v = np.r_[v,stream_velocity]
        deposited = np.r_[deposited,np.zeros(count,bool)]
        start = 0.

    seen, held = [], []
    for step in range(481):
        t = step*dt
        if step%10 == 0:
            seen.append(x.copy())
            held.append(deposited.copy())
        if step==480:
            break
        step_dt = np.full(len(x),dt)
        if start<=t<stop:
            onset = min((t-start)/.15,1.) if not row.get('already_pouring',False) else 1.
            onset = min(onset,(stop-t)/.13)
            # Stagger births over the solver step. Synchronized discs create
            # visible gaps along a long or fast jet, especially after projection.
            count = max(1, round(.5*flow_count*onset))
            fresh = emit(t,count)
            # Each birth is at an independent time inside the integration step.
            # This preserves even emission as the jet accelerates under gravity.
            step_dt = np.r_[step_dt,rng.uniform(.02*dt,dt,count)]
            x = np.r_[x, fresh]
            v = np.r_[v, np.tile(velocity, (count,1))]
            deposited = np.r_[deposited, np.zeros(count,bool)]
        if not len(x):
            continue
        old = x.copy()
        v[:, 1] -= 9.81*step_dt
        candidate = old + step_dt[:,None]*v
        candidate[~deposited,1] += .5*9.81*step_dt[~deposited]**2
        crossed = (~deposited) & (old[:,1]>=height) & (candidate[:,1]<height)
        crossed_ids = np.flatnonzero(crossed)
        before,after = old[crossed_ids],candidate[crossed_ids]
        fraction = (before[:,1]-height)/(before[:,1]-after[:,1])
        intersection = before+fraction[:,None]*(after-before)
        through_opening = np.linalg.norm(intersection[:,(0,2)],axis=1)<cavity_radius(height)
        deposited[crossed_ids[through_opening]] = True
        candidate = contacts(candidate, deposited)
        # Exclude fluid elements outside the receiving vessel from boundary
        # projection. The thin free jet follows ballistic advection until its
        # particles cross the open rim. Density projection acts on the
        # collected liquid, where an incompressible bulk volume is resolved.
        for _ in range(3):
            collected = np.flatnonzero(deposited)
            pairs = cKDTree(candidate[collected]).query_pairs(kernel,output_type='ndarray')
            i,j = collected[pairs].T
            diff = candidate[i]-candidate[j]
            dist = np.linalg.norm(diff,axis=1)
            good = dist>1e-7
            i,j,diff,dist = i[good],j[good],diff[good],dist[good]
            w = poly6*(kernel**2-dist**2)**3
            rho = np.full(len(candidate),mass*poly6*kernel**6)
            np.add.at(rho,i,mass*w)
            np.add.at(rho,j,mass*w)
            grad = mass/1000*spiky*(kernel-dist)[:,None]**2*diff/dist[:,None]
            sums = np.zeros_like(candidate)
            np.add.at(sums,i,grad)
            np.add.at(sums,j,-grad)
            ghost_dist = 2*(candidate[:,1]-floor)
            ghost = deposited & (ghost_dist>0) & (ghost_dist<kernel)
            rho[ghost] += mass*poly6*(kernel**2-ghost_dist[ghost]**2)**3
            ghost_grad = np.zeros_like(candidate)
            ghost_grad[ghost,1] = mass/1000*spiky*(kernel-ghost_dist[ghost])**2
            sums += ghost_grad
            norm = np.sum(sums**2+ghost_grad**2,axis=1)
            pair_norm = np.sum(grad**2,axis=1)
            np.add.at(norm,i,pair_norm)
            np.add.at(norm,j,pair_norm)
            lam = -np.maximum(rho/1000-1,0)/(norm+200.)
            correction = (lam[i]+lam[j])[:,None]*grad
            displacement = lam[:,None]*ghost_grad
            np.add.at(displacement,i,correction)
            np.add.at(displacement,j,-correction)
            candidate = contacts(candidate+np.clip(displacement,-spacing*.3,spacing*.3),deposited)
        v[deposited] = .998*(candidate[deposited]-old[deposited])/step_dt[deposited,None]
        smoother = np.zeros_like(v)
        delta = .042*(1-dist/kernel)[:,None]**3*(v[j]-v[i])
        np.add.at(smoother,i,delta)
        np.add.at(smoother,j,-delta)
        v += smoother
        x = candidate
    return seen,held,spacing


def jet_surface(points, spacing, nominal_radius, nozzle, velocity):
    """Reconstruct connected jet cross-sections from dense in-flight particles."""
    if len(points)<12:
        return None
    points = points[(points[:,1] <= nozzle[1]+spacing) &
                    (points[:,1] >= -spacing)]
    heights = np.floor(points[:,1]/spacing).astype(int)
    slices = [(y, points[heights==y]) for y in np.unique(heights)]
    slices = [p for _,p in slices if len(p)>=2]
    if len(slices)<3:
        return None
    centers = np.array([np.median(p,axis=0) for p in slices])
    # Only complete stretches supported by nearby occupied cross-sections.
    connected = np.diff(centers[:,1])<=3.5*spacing
    radii = np.array([np.clip(np.quantile(np.linalg.norm(p[:,(0,2)]-c[[0,2]],axis=1),.70)+.0008,
                              .48*nominal_radius,1.30*nominal_radius) for p,c in zip(slices,centers)])
    # Remove finite-sample jitter across neighboring slices around the same
    # ballistic stream. Retain coherent deflections and genuine empty gaps.
    age = (velocity[1]+np.sqrt(velocity[1]**2+2*9.81*np.maximum(0,nozzle[1]-centers[:,1])))/9.81
    expected = nozzle[[0,2]]+age[:,None]*velocity[[0,2]]
    residual = centers[:,(0,2)]-expected
    for group in np.split(np.arange(len(centers)),np.flatnonzero(~connected)+1):
        centers[np.ix_(group,[0,2])] = expected[group]+gaussian_filter1d(residual[group],1.7,axis=0,mode='nearest')
        radii[group] = gaussian_filter1d(radii[group],1.2,mode='nearest')
    ring_size = 12
    angles = np.linspace(0,2*np.pi,ring_size,endpoint=False)
    ring = np.c_[np.cos(angles),np.zeros(ring_size),np.sin(angles)]
    verts = (centers[:,None,:]+radii[:,None,None]*ring).reshape(-1,3)
    faces = []
    for y in np.flatnonzero(connected):
        for j in range(ring_size):
            a = y*ring_size+j
            b = y*ring_size+(j+1)%ring_size
            faces.extend([(a,b,a+ring_size),(b,b+ring_size,a+ring_size)])
    return (verts,np.asarray(faces,dtype=np.int64)) if faces else None


def run(row, output_root):
    out = Path(output_root)/row['case']
    out.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(row['image'])
    if image is None or image.shape[:2]!=(256,384):
        raise ValueError('Liquid landmarks require an unaltered 384x256 input canvas')
    content_rows = np.flatnonzero(np.max(image,axis=(1,2))>3)
    first_content,last_content = content_rows[[0,-1]]
    verts, faces, rotation, offset, K, radius, width_profile, floor, height, error = fit_vessel(
        row['rim'],row['base'],row.get('middle_width'))
    nozzle, velocity = activate(row,rotation,offset,K,height,radius)
    positions, capture, spacing = simulate(row,radius,width_profile,floor,height,nozzle,velocity)
    from scipy.ndimage import minimum_filter
    transform = lambda xyz: xyz@rotation.T+offset
    surfaces,triangles = transform(verts),faces
    if 'handle_box' in row:
        loop,loop_faces = handle_surface(row['handle_box'],offset[2],K)
        triangles = np.r_[triangles,loop_faces+len(surfaces)]
        surfaces = np.r_[surfaces,loop]
    cup = farthest(mesh_surface_candidates(surfaces,triangles,count=11000,seed=51),row.get('vessel_points',800))
    receiver_depth = raster_accelerated(surfaces,triangles,K,(256,384))
    near = float(np.min(cup[:,2])-.02)
    far = float(np.max(cup[:,2])+.08)
    solids = np.where(receiver_depth>0,receiver_depth,np.inf)
    ps,ds,physical_depths = [],[],[]
    fluid_ids = np.random.default_rng(873).random(len(positions[-1]))<row.get('keep_fraction',.72)
    for x,held in zip(positions,capture,strict=True):
        fluid = transform(x)
        count = len(fluid)
        visible = np.r_[cup,fluid[fluid_ids[:count] & held]]
        q = visible@K.T
        xy = q[:,:2]/q[:,2:]
        p,_ = raster_subpixel_points(xy,visible[:,2],visible[:,2]>.001,384,256,near,far,radius=.83)
        z = solids.copy()
        if count:
            q = fluid@K.T
            jet = ~held
            stream = fluid[jet]
            jet_pixels = q[jet,:2]/q[jet,2:]
            jet_particle,_ = raster_subpixel_points(jet_pixels,stream[:,2],stream[:,2]>.001,
                                                    384,256,near,far,radius=.98)
            p = np.maximum(p,jet_particle)
            uv = np.rint(q[:,:2]/q[:,2:]).astype(int)
            valid = (uv[:,0]>=0)&(uv[:,0]<384)&(uv[:,1]>=0)&(uv[:,1]<256)&(fluid[:,2]>.001)
            fitted_jet = jet_surface(x[jet],spacing,row.get('nozzle_radius',.0042),nozzle,velocity)
            if fitted_jet is not None:
                jet_vertices,jet_faces = fitted_jet
                jet_depth = raster_accelerated(transform(jet_vertices),jet_faces,K,(256,384))
                z = np.minimum(z,np.where(jet_depth>0,jet_depth,np.inf))
                valid &= held
            uv = uv[valid]
            surf = np.full_like(z,np.inf)
            np.minimum.at(surf,(uv[:,1],uv[:,0]),fluid[valid,2]-spacing/2)
            # Approximate the first surface from dense native samples,
            # independently of the sparse selection used for particles.
            extent = max(3, 2*round(.45*spacing*K[0,0]/np.median(fluid[valid,2]))+1) if valid.any() else 3
            z = np.minimum(z,minimum_filter(surf,size=extent))
        d = np.where(np.isfinite(z), np.clip(.57+.40*(far-z)/(far-near),.25,.99), 0)
        p[:first_content] = 0
        p[last_content+1:] = 0
        d[:first_content] = 0
        d[last_content+1:] = 0
        z[:first_content] = np.inf
        z[last_content+1:] = np.inf
        physical_depths.append(np.where(np.isfinite(z),z,0).astype(np.float32))
        ps.append(p)
        ds.append(np.repeat(np.rint(d*255).astype(np.uint8)[...,None],3,axis=2))
    video(out/'particles.mp4',np.asarray(ps))
    video(out/'depth.mp4',np.asarray(ds))
    video(out/'overview.mp4',np.asarray([np.hstack((image,p,d)) for p,d in zip(ps,ds,strict=True)]))
    cv2.imwrite(str(out/'first_frame.png'),image)
    dense = np.full((len(positions),len(positions[-1]),3),np.nan,dtype=np.float32)
    held = np.zeros(dense.shape[:2],bool)
    for f,(points,inside) in enumerate(zip(positions,capture,strict=True)):
        dense[f,:len(points)] = points
        held[f,:len(points)] = inside
    np.savez_compressed(out/'states.npz',positions=dense,captured=held,selected_ids=np.flatnonzero(fluid_ids),
                        cup=cup,K=K,rotation=rotation,offset=offset,times_s=np.arange(49)/12,
                        depth_z=physical_depths,cup_vertices_camera=surfaces,cup_faces=triangles)
    captured = capture[-1]
    last = positions[-1]
    inside = captured & (last[:,1]<=height) & (last[:,1]>=floor)
    liquid = last[inside]
    retained = float(captured.mean()) if len(captured) else 0.
    settled = float(inside.mean()) if len(inside) else 0.
    checks = {'Rim or foot fit exceeds 5 pixels':bool(error<5),
              'Less than 97% of the liquid ends inside the vessel':bool(settled>=.97),
              'Liquid state is non-finite':all(bool(np.isfinite(frame).all()) for frame in positions),
              'Liquid never enters the receiver':bool(len(liquid) and len(captured)>len(positions[0])),
              'Initial stream is absent':bool(not row.get('already_pouring') or (~capture[0]).any())}
    if row.get('liquid_stop_condition') == 'before_rim':
        checks['Liquid did not settle below the rim after the pour stopped'] = bool(
            len(last) and np.max(last[:, 1]) < height and captured.all())
    report = dict(case=row['case'],image_sha256=digest(row['image']),prompt=row['prompt'],
                  backend='CPU position-based liquid',landmark_rmse=error,
                  liquid_initial_count=len(positions[0]),initial_in_flight_count=int((~capture[0]).sum()),
                  liquid_final_count=len(captured),
                  final_captured_fraction=retained,final_settled_fraction=settled,
                  peak_liquid_height_m=float(liquid[:,1].max()) if len(liquid) else None,
                  vessel_rim_radius_m=radius,vessel_rim_height_m=height,
                  particles_per_emission=row.get('particles_per_emission',14),
                  stop_condition=row.get('liquid_stop_condition'),
                  volume_spacing_m=spacing,frames=49,fps=12,
                  projection_scope='vessel and incoming liquid',nozzle_world=nozzle.tolist(),
                  assumptions=row.get('assumptions',[]),checks=checks,numeric_pass=all(checks.values()),
                  visual_pass=None)
    (out/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cv2.setNumThreads(1)
    for row in json.loads(args.manifest.read_text()):
        print(json.dumps(run(row,args.output)),flush=True)


if __name__ == '__main__':
    main()
