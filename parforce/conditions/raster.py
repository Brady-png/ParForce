"""Project CPU rasterizer, preserving camera pixel centers and perspective depth."""
import numpy as np
from numba import njit
def project(vertices,K):
 q=vertices@K.T
 return q[:,:2]/q[:,2,None]

def _raster_kernel(vertices,faces,uv,h,w):
    depth=np.full((h,w),np.inf,np.float32)
    for f in range(len(faces)):
        ia,ib,ic=faces[f,0],faces[f,1],faces[f,2]
        za,zb,zc=vertices[ia,2],vertices[ib,2],vertices[ic,2]
        if za<=.001 or zb<=.001 or zc<=.001:continue
        ax,ay=uv[ia,0],uv[ia,1];bx,by=uv[ib,0],uv[ib,1];cx,cy=uv[ic,0],uv[ic,1]
        lx=max(int(np.floor(min(ax,bx,cx))),0);ly=max(int(np.floor(min(ay,by,cy))),0)
        hx=min(int(np.ceil(max(ax,bx,cx))),w-1);hy=min(int(np.ceil(max(ay,by,cy))),h-1)
        den=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
        if abs(den)<1e-10:continue
        for y in range(ly,hy+1):
            for x in range(lx,hx+1):
                px=x+.5;py=y+.5
                w0=((by-cy)*(px-cx)+(cx-bx)*(py-cy))/den
                w1=((cy-ay)*(px-cx)+(ax-cx)*(py-cy))/den;w2=1-w0-w1
                if w0>=-1e-6 and w1>=-1e-6 and w2>=-1e-6:
                    inv=w0/za+w1/zb+w2/zc
                    if inv>0:
                        z=1/inv
                        if z<depth[y,x]:depth[y,x]=z
    for y in range(h):
        for x in range(w):
            if not np.isfinite(depth[y,x]):depth[y,x]=0
    return depth


if njit is not None:
    _raster_kernel=njit(cache=False,fastmath=False,boundscheck=True)(_raster_kernel)


def raster_accelerated(vertices,faces,K,shape=(256,384)):
    """Same perspective/pixel-center contract; optional in-memory CPU JIT only."""
    return _raster_kernel(np.asarray(vertices),np.asarray(faces),project(vertices,K),shape[0],shape[1])
