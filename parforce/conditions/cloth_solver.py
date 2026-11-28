"""Position-based cloth constraints and anchored wind rollout."""

from numba import njit
import numpy as np
from scipy.spatial import cKDTree


def edges_and_bends(faces, initial):
    edges={}
    for a,b,c in faces:
        for i,j,k in ((a,b,c),(a,c,b),(b,c,a)):
            edges.setdefault(tuple(sorted((i,j))),[]).append(k)
    bends={tuple(sorted(v)) for v in edges.values() if len(v)==2 and v[0]!=v[1]}
    def constraints(pairs):
        ij=np.asarray(sorted(pairs),dtype=np.int32)
        rest=np.linalg.norm(initial[ij[:,0]]-initial[ij[:,1]],axis=1)
        valid=rest>1e-6
        return ij[valid],rest[valid]
    return constraints(edges),constraints(bends)

@njit(cache=True)
def constrain(x, inv_mass, pairs, rest, impulse, compliance, unilateral=False):
    for k in range(len(pairs)):
        i,j=pairs[k]
        dx=x[i,0]-x[j,0]
        dy=x[i,1]-x[j,1]
        dz=x[i,2]-x[j,2]
        dist=np.sqrt(dx*dx+dy*dy+dz*dz)
        denom=inv_mass[i]+inv_mass[j]+compliance
        if dist>1e-10 and denom>0 and (not unilateral or dist<rest[k]):
            amount=(-(dist-rest[k])-compliance*impulse[k])/denom
            impulse[k]+=amount
            s=amount/dist
            for axis, delta in enumerate((dx,dy,dz)):
                x[i,axis]+=inv_mass[i]*s*delta
                x[j,axis]-=inv_mass[j]*s*delta

def rollout(initial, faces, grip, fixed, travel, driven, wind=None, gravity_z=-9.81,
            lift_height=.14, grip_start=.22, slide_start=.85, return_control=False,
            hold_until_grip=False, ramp_scale=1., loaded_rest_rate=0., gust_end=None):
    (edges,lengths),(bends,bend_lengths)=edges_and_bends(faces,initial)
    dt=1/144
    positions=initial.copy()
    velocities=np.zeros_like(positions)
    response=[positions.copy()]
    inv=np.ones(len(initial))
    inv[fixed]=0.
    use_grip=driven and len(grip)>0
    if use_grip:
        inv[grip]=0.
    links={tuple(edge) for edge in edges}
    links.update(map(tuple,bends))
    floor=.002
    gravity=np.array([0.,0.,gravity_z])
    height=np.ptp(initial[:,2])
    def eased(x):
        x=float(np.clip(x,0,1))
        return x*x*(3-2*x)
    for step in range(1,577):
        t=step*dt
        onset=min(grip_start,slide_start) if use_grip else .18
        if hold_until_grip and driven and t<=onset:
            if step%12==0:
                response.append(initial.copy())
            continue
        last=positions.copy()
        acceleration=gravity
        if driven and wind is not None:
            relative_depth=(np.max(initial[:,2])-initial[:,2])/height
            amplitude=np.maximum(0,1-.3*relative_depth)*eased((t-.18)/(.4*ramp_scale))
            amplitude*=.65+.35*np.sin(2*np.pi*(.65*t-.4*relative_depth))**2
            if gust_end is not None:
                amplitude*=1-eased((t-gust_end)/.50)
            # A photographed curtain is already loaded by gravity. When modeled
            # around that equilibrium, retain a restoring response to deflection.
            acceleration=gravity+amplitude[:,None]*wind-loaded_rest_rate**2*(positions-initial)
        positions+=(velocities*.965+dt*acceleration)*dt*inv[:,None]
        qa=np.zeros(len(edges)); qb=np.zeros(len(bends))
        if use_grip:
            lift=eased((t-grip_start)/(.63*ramp_scale))
            slide=eased((t-slide_start)/(1.25*ramp_scale))
            grip_pos=initial[grip]+slide*travel+np.array([0,0,lift_height*lift])
        for iteration in range(10):
            if use_grip:
                positions[grip]=grip_pos
            positions[fixed]=initial[fixed]
            constrain(positions,inv,edges,lengths,qa,.0008)
            constrain(positions,inv,bends,bend_lengths,qb,.16)
            if iteration in (3,7):
                neighbors=cKDTree(positions).query_pairs(.0032,output_type="ndarray")
                pairs=np.asarray([pair for pair in neighbors if tuple(pair) not in links],dtype=np.int32).reshape(-1,2)
                minimum=np.full(len(pairs),.0032)
                constrain(positions,inv,pairs,minimum,np.zeros(len(pairs)),.005,True)
            positions[:,2]=np.maximum(positions[:,2],floor)
        velocities=(positions-last)/dt
        velocities[positions[:,2]<floor+.00001,:2]*=.65
        if step%12==0:
            response.append(positions.copy())
    if return_control:
        slide=np.asarray([eased((t-slide_start)/(1.25*ramp_scale)) for t in TIMES])
        lift=np.asarray([eased((t-grip_start)/(.63*ramp_scale)) for t in TIMES])
        return np.asarray(response),edges,lengths,slide[:,None]*travel+lift[:,None]*[0,0,lift_height]
    return np.asarray(response),edges,lengths
