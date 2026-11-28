"""Surface sampling and fractional-radius particle rendering."""

import cv2
import numpy as np


def mesh_surface_candidates(vertices,faces,count=4000,seed=31):
 rng=np.random.default_rng(seed);tri=vertices[faces]
 area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
 chosen=rng.choice(len(faces),count,p=area/area.sum());uv=rng.random((count,2));uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1]
 return tri[chosen,0]+uv[:,:1]*(tri[chosen,1]-tri[chosen,0])+uv[:,1:]*(tri[chosen,2]-tri[chosen,0])

def farthest(points,count):
 selected=[];d=np.full(len(points),np.inf);i=0
 for _ in range(count):
  selected.append(i);d=np.minimum(d,np.sum((points-points[i])**2,axis=1));i=int(d.argmax())
 return points[selected]

def depth_intensity(z,near,far):
 t=np.clip((np.asarray(z)-near)/max(far-near,1e-8),0,1)
 return np.rint(255-165*t).astype(np.uint8)

def raster_subpixel_points(uv, z, valid, w, h, near, far, radius=.35):
    """8x area-sampled disks with fractional centers/radii; no disk dilation.

    Fixed-point drawing preserves subpixel radius instead of rounding the
    requested source radius to a whole supersampled pixel. Area downsampling
    supplies the antialiasing; the large footprints of LINE_AA are not added.
    """
    if radius <= 0 or not np.isfinite(radius): raise ValueError('Radius must be finite and positive')
    valid = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    ids = np.flatnonzero(valid); ids = ids[np.argsort(z[ids], kind='stable')[::-1]]
    canvas = np.zeros((h*8, w*8), np.uint8)
    colors = depth_intensity(z, near, far)
    centers = np.rint(((uv[ids]+.5)*8-.5)*256).astype(np.int64)
    fixed_radius = max(1, int(round(radius*8*256)))
    for index, center in zip(ids, centers):
        cv2.circle(canvas, tuple(map(int, center)), fixed_radius, int(colors[index]),
                   -1, lineType=cv2.LINE_8, shift=8)
    frame = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
    return np.repeat(frame[..., None], 3, -1), int(valid.sum())
