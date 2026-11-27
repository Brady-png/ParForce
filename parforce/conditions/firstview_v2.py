"""First-view anchored surfaces and explicit support graphs for condition pilots.

Hidden back surfaces and coarse collision volumes are declared priors, not measured shape.
This module accepts instance annotations; it has no case-ID branches.
"""
from pathlib import Path
import json,hashlib,sys,os
import cv2,numpy as np,trimesh,mujoco
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

TIMES=np.arange(49)/12

def digest(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()

def clean_depth(depth):
 valid=np.isfinite(depth)&(depth>0)
 if not valid.any():raise ValueError('No valid source depth')
 ix=distance_transform_edt(~valid,return_distances=False,return_indices=True)
 return depth[tuple(ix)].astype(np.float64)

def backproject(depth,K):
 y,x=np.indices(depth.shape);return np.stack((x+.5,y+.5,np.ones_like(x)),axis=-1)@np.linalg.inv(K).T*depth[...,None]

def basis_from_up(up):
 up=np.asarray(up,float);up/=np.linalg.norm(up);right=np.array([1.,0.,0.]);right-=up*(up@right);right/=np.linalg.norm(right)
 return np.column_stack((right,np.cross(up,right),up))

def infer_up(cloud,masks,actors):
 centers=np.array([np.median(cloud[m],0)for m in masks]);directions=[]
 for i,a in enumerate(actors):
  support=a.get('support_actor')
  if support is not None:
   ids=support if isinstance(support,list)else[support];v=centers[i]-centers[ids].mean(0)
   if v[1]<0 and np.linalg.norm(v)>1e-5:directions.append(v/np.linalg.norm(v))
 occupied=np.any(masks,0);roots=[i for i,a in enumerate(actors)if a.get('support_actor')is None];root=np.any(masks[roots],0);y,x=np.where(root);h,w=root.shape;zone=np.zeros_like(root);bottom=int(np.quantile(y,.985));zone[max(0,bottom+2):min(h,bottom+55),max(0,int(x.min()-35)):min(w,int(x.max()+35))]=True
 eligible=zone&~cv2.dilate(occupied.astype('uint8'),np.ones((9,9),'uint8')).astype(bool);p=cloud[eligible][::2]
 normal=None;relative_residual=None
 if len(p)>70:
  original=p.copy()
  for _ in range(4):
   c=np.median(p,0);n=np.linalg.svd(p-c,full_matrices=False)[2][-1];err=np.abs((p-c)@n);p=p[err<=np.quantile(err,.8)]
  if n[1]>0:n=-n
  relative_residual=float(np.median(np.abs((original-c)@n))/max(1e-8,np.ptp(original,axis=0).max()))
  normal=n
 upright_hint=any(a.get('upright_hint') or any(t in a.get('name','').lower()for t in ['upright','domino','standing pin']) or a.get('primitive')=='pin' for a in actors)
 if directions or upright_hint:
  up=np.array([0.,-1.,0.]);source='camera-roll-zero upright-axis prior from supported-object layout; local terrain normal is not equated with gravity'
 else:
  if normal is None:
   eligible=~cv2.dilate(occupied.astype('uint8'),np.ones((15,15),'uint8')).astype(bool);p=cloud[eligible][::5];c=np.median(p,0);normal=np.linalg.svd(p-c,full_matrices=False)[2][-1]
  up=normal if normal[2]<0 else-normal;source='observed common support patch for non-upright layout'
 return up,dict(source=source,up=up.tolist(),support_plane_residual=relative_residual)

def observed_mesh(depth,K,mask,step=2,closed=True,thickness_fraction=.28):
 """Preserve observed front surface; close only an explicitly solid actor's unseen side."""
 valid=mask&(depth>0);nearest=distance_transform_edt(~valid,return_distances=False,return_indices=True)
 local=depth[tuple(nearest)].astype(np.float32);local=cv2.bilateralFilter(local,5,float(np.median(local[mask])*.015),3)
 yy,xx=np.mgrid[0:mask.shape[0]:step,0:mask.shape[1]:step];idx=np.arange(xx.size).reshape(xx.shape)
 a,b,c,d=idx[:-1,:-1].ravel(),idx[:-1,1:].ravel(),idx[1:,:-1].ravel(),idx[1:,1:].ravel();faces=np.concatenate((np.c_[a,c,b],np.c_[b,c,d]));keep=mask[yy,xx].ravel();faces=faces[keep[faces].all(1)]
 rays=np.c_[xx.ravel()+.5,yy.ravel()+.5,np.ones(xx.size)]@np.linalg.inv(K).T;verts=rays*local[yy,xx].ravel()[:,None]
 mesh=trimesh.Trimesh(verts,faces,process=False);mesh.remove_unreferenced_vertices()
 if len(mesh.faces)<20:raise ValueError('Insufficient observed surface, refine mask or sampling step')
 if closed:
  n=len(mesh.vertices);counts=np.bincount(mesh.edges_unique_inverse);edge=mesh.edges_unique[counts==1];v=mesh.vertices.copy()
  # Ray extrusion preserves the observed silhouette. Hidden thickness is bounded
  # by projected extent and recorded; it never changes across animation frames.
  span=np.ptp(v,axis=0);thickness=max((span[:2].min() if thickness_fraction>=.65 else span[:2].max())*thickness_fraction,np.median(v[:,2])*.002)
  back=v*((v[:,2]+thickness)/v[:,2])[:,None]
  sides=np.concatenate((np.c_[edge[:,0],edge[:,1],edge[:,0]+n],np.c_[edge[:,1],edge[:,1]+n,edge[:,0]+n]))
  mesh=trimesh.Trimesh(np.r_[v,back],np.r_[mesh.faces,mesh.faces[:,::-1]+n,sides],process=False)
  mesh.metadata.update(hidden_completion='bounded camera-ray extrusion; unobserved backside prior',hidden_thickness=float(thickness),front_vertex_count=n)
 return mesh

def observed_sphere(depth,K,mask):
 y,x=np.where(mask);u,v=np.mean(x)+.5,np.mean(y)+.5;pixel_radius=np.sqrt(len(x)/np.pi);ratio=pixel_radius/K[0,0]
 cz=float(np.median(depth[mask]))/max(.2,1-.7*ratio);radius=ratio*cz;center=np.linalg.inv(K)@np.array([u,v,1.])*cz
 mesh=trimesh.creation.icosphere(subdivisions=4,radius=radius);mesh.apply_translation(center);mesh.metadata.update(hidden_completion='complete analytic sphere fitted to observed silhouette and monocular scale prior');return mesh

def background_depth(depth,masks,protect=None):
 """Remove actor-owned depth without copying its previous silhouette to the background."""
 from scipy.sparse import coo_matrix
 from scipy.sparse.linalg import spsolve
 mask=cv2.dilate(np.any(masks,0).astype('uint8'),np.ones((25,25),'uint8')).astype(bool)
 mask=mask&~(np.asarray(protect,bool)&~np.any(masks,0))if protect is not None else mask
 inv=(1/depth).astype(np.float64);yy,xx=np.where(mask);idx=np.full(mask.shape,-1,int);idx[yy,xx]=np.arange(len(yy));rr=[];cc=[];vv=[];rhs=np.zeros(len(yy));degree=np.zeros(len(yy))
 for dy,dx in [(0,-1),(0,1),(-1,0),(1,0)]:
  yn,xn=yy+dy,xx+dx;valid=(yn>=0)&(yn<mask.shape[0])&(xn>=0)&(xn<mask.shape[1]);r=np.flatnonzero(valid);yn,xn=yn[valid],xn[valid];degree[r]+=1;unknown=mask[yn,xn];rr.extend(r[unknown]);cc.extend(idx[yn[unknown],xn[unknown]]);vv.extend(-np.ones(unknown.sum()));rhs[r[~unknown]]+=inv[yn[~unknown],xn[~unknown]]
 rr.extend(np.arange(len(yy)));cc.extend(np.arange(len(yy)));vv.extend(degree);matrix=coo_matrix((vv,(rr,cc)),shape=(len(yy),len(yy))).tocsr();sol=spsolve(matrix,rhs);out=inv.copy();out[yy,xx]=sol;return 1/np.maximum(out,1e-6)

def disjoint_masks(masks):
 masks=np.asarray(masks,bool);scores=np.array([distance_transform_edt(m)for m in masks]);winner=scores.argmax(0);return masks&(winner[None]==np.arange(len(masks))[:,None,None])

def calibrate_root_depths(meshes,actors,basis):
 roots=[i for i,a in enumerate(actors)if a.get('support_actor')is None and a.get('motion_role')!='static_background'];heights=np.array([(m.vertices@basis)[:,2].min()for m in meshes]);floor=float(np.median(heights[roots]));records=[]
 for i in roots:
  ratio=floor/heights[i]if abs(heights[i])>1e-6 else 1.
  if ratio<=0 or not .6<=ratio<=1.7:raise ValueError('Root depths inconsistent with one observed support; revisit camera/support perception')
  meshes[i].vertices*=ratio;records.append(dict(actor=i,ray_scale=float(ratio)))
 return records

def complete_shared_hidden_support(meshes,actors):
 indices=[i for i,a in enumerate(actors)if a.get('support_actor')is not None or any(i in (b.get('support_actor')if isinstance(b.get('support_actor'),list)else[b.get('support_actor')])for b in actors)]
 solids=[i for i in indices if meshes[i].metadata.get('front_vertex_count')]
 if len(solids)<2:return None
 front=[meshes[i].vertices[:meshes[i].metadata['front_vertex_count']]for i in solids];back_z=max(np.quantile(v[:,2],.98)for v in front)+.45*np.median([np.ptp(v[:,0])for v in front])
 for i in solids:
  m=meshes[i];n=m.metadata['front_vertex_count'];m.vertices[n:]=m.vertices[:n]*(back_z/m.vertices[:n,2])[:,None];m.metadata.update(hidden_completion='shared hidden back-depth prior for a support-connected solid group; observed fronts unchanged',shared_back_depth=float(back_z))
 return float(back_z)

def contact_bounds(vertices,actors,basis):
 world=[v@basis for v in vertices];centers=np.array([(v.min(0)+v.max(0))/2 for v in world]);bounds=np.array([[v.min(0),v.max(0)]for v in world]);original=bounds.copy();interfaces=[]
 # Solve shared interfaces jointly: processing two children sequentially must
 # not move a support face twice and leave its earlier child penetrating.
 roots=[i for i,a in enumerate(actors)if a.get('support_actor')is None]
 floor=float(np.median(bounds[roots,0,2]));constraints=[];rhs=[]
 for i in roots:
  eq=np.zeros(2*len(actors));eq[2*i]=1;constraints.append(eq);rhs.append(floor)
 for i,a in enumerate(actors):
  support=a.get('support_actor')
  if support is None:continue
  ids=support if isinstance(support,list)else[support]
  for j in ids:
   eq=np.zeros(2*len(actors));eq[2*i]=1;eq[2*j+1]=-1;constraints.append(eq);rhs.append(0.)
  interfaces.append(dict(child=i,supporters=ids))
 A=np.array(constraints);b=np.array(rhs);observed=bounds[:,:,2].reshape(-1)
 fitted=observed-A.T@np.linalg.lstsq(A@A.T,A@observed-b,rcond=None)[0]
 if np.max(np.abs(A@fitted-b))>1e-7:raise ValueError('Inconsistent support graph')
 bounds[:,:,2]=fitted.reshape(-1,2)
 for entry in interfaces:entry['contact_height']=float(bounds[entry['child'],0,2])
 if np.any(bounds[:,1]-bounds[:,0]<=0):raise ValueError('Invalid support graph or shape bounds')
 adjustment=np.abs(bounds-original).max(axis=(1,2))/np.maximum(np.ptp(original,axis=1).max(1),1e-8)
 if adjustment.max()>.35:raise ValueError(f'Support assembly exceeds visible shape uncertainty: {adjustment.tolist()}')
 return world,centers,bounds,floor,dict(interfaces=interfaces,proxy_bound_adjustment_fraction=adjustment.tolist(),floor_height=floor)

def rigid_rollout(meshes,actors,basis,out,axis=-1.,force_gain=1.,contact_target_actor=None,direction_camera=None,actuation_type='push',force_height_fraction=0.,motion_scale=1.):
 world,centers,bounds,floor,report=contact_bounds([m.vertices for m in meshes],actors,basis);bodies=[];static=[]
 for i,(m,a,center,bb)in enumerate(zip(meshes,actors,centers,bounds)):
  extent=bb[1]-bb[0];offset=(bb[1]+bb[0])/2-center;size=extent/2;kind=a.get('primitive');fixed=a.get('motion_role')=='fixed_support'
  if fixed:static.append(i)
  if kind=='sphere':
   radius=float(np.min(m.extents)/2);shape=f'type="sphere" size="{radius}"';offset[:]=0;offset[2]=bb[0,2]+radius-center[2]
  elif kind=='box' or any(a.get('support_actor')is not None for a in actors):shape=f'type="box" size="{fmt(size)}"'
  else:
   from scipy.spatial import ConvexHull
   has_support_links=any(a.get('support_actor')is not None for a in actors)
   if has_support_links:
    xy=np.array([[bb[0,0],bb[0,1]],[bb[1,0],bb[0,1]],[bb[1,0],bb[1,1]],[bb[0,0],bb[1,1]]])
   else:xy=world[i][ConvexHull(world[i][:,:2]).vertices,:2]
   verts=np.r_[np.c_[xy,np.full(len(xy),bb[0,2])],np.c_[xy,np.full(len(xy),bb[1,2])]]
   proxy=trimesh.convex.convex_hull(verts-center);proxy.export(out/f'collider_{i:02d}.obj');shape=f'type="mesh" mesh="proxy_{i}"';offset[:]=0
  joint=''if fixed else'<freejoint/>'
  mass=max(1e-5,float(np.prod(extent))*450)
  bodies.append(f'<body name="body_{i}" pos="{fmt(center)}">{joint}<geom name="actor_{i}" {shape} pos="{fmt(offset)}" mass="{mass}"/></body>')
 assets='<asset>'+''.join(f'<mesh name="proxy_{i}" file="{out/f"collider_{i:02d}.obj"}"/>'for i,a in enumerate(actors)if a.get('primitive')not in ['sphere','box']and not any(b.get('support_actor')is not None for b in actors))+'</asset>'
 xml=f'<mujoco>{assets}<option timestep="0.001041666666666667" integrator="implicitfast" iterations="100" gravity="0 0 -9.81"/><default><geom friction=".55 .01 .002" condim="6" solref=".004 1" solimp=".99 .999 .0001"/></default><worldbody><geom name="observed_support_plane_prior" type="plane" pos="0 0 {floor}" size="5 5 .01"/>'+''.join(bodies)+'</worldbody></mujoco>'
 (out/'scene.xml').write_text(xml);model=mujoco.MjModel.from_xml_string(xml);target=0;descendants={target}
 for _ in actors:
  for i,a in enumerate(actors):
   s=a.get('support_actor');ids=s if isinstance(s,list)else[s]
   if s is not None and any(j in descendants for j in ids):descendants.add(i)
 mass=float(model.body_mass[1]);load=float(sum(model.body_mass[i+1]for i in descendants));width=float(meshes[0].extents[0]);duration=.6;peak=(1.12*.55*9.81*(2*load-mass)+mass*1.5*width/duration**2)*force_gain
 direction=np.array([axis,0.,0.])
 if direction_camera is not None:
  direction=basis.T@np.asarray(direction_camera,float);direction[2]=0;direction/=max(np.linalg.norm(direction),1e-8)
 elif contact_target_actor is not None:
  direction=centers[contact_target_actor]-centers[0];direction[2]=0;direction/=np.linalg.norm(direction)
 def simulate(active):
  data=mujoco.MjData(model);mujoco.mj_forward(model,data);poses=[];pen=0.;vel=[];contacted=False
  for t in TIMES:
   if t<=.25 and actuation_type!='gravity':data.time=float(t)
   while data.time<t-1e-8:
    drive_duration=2.5 if contact_target_actor is not None else 1.25;data.xfrc_applied[:]=0;p=(data.time-.25)/drive_duration
    if active and actuation_type!='gravity' and 0<p<1:
     touching=False
     if contact_target_actor is not None:
      for c in data.contact[:data.ncon]:
       ids={int(model.geom_bodyid[c.geom1])-1,int(model.geom_bodyid[c.geom2])-1}
       if ids=={0,contact_target_actor}and data.time>.55:touching=True;contacted=True
     if not contacted:
      speed=np.dot(data.qvel[:3],direction);desired=.32*width*motion_scale
      if contact_target_actor is not None:
       distance=np.linalg.norm(centers[contact_target_actor,:2]-centers[0,:2]);desired=min(1.2*width*motion_scale,max(desired,.65*distance*motion_scale))
      ease=min(1.,p/.16,(1-p)/.16);gate=np.clip((1.25*desired-speed)/max(.25*desired,1e-6),0,1);magnitude=gate*np.clip(.55*9.81*(2*load-mass)+mass*10*(desired-speed),0,peak)
      data.xfrc_applied[1,:3]=ease*magnitude*direction
      if force_height_fraction:
       arm=np.array([0.,0.,force_height_fraction*(bounds[0,1,2]-bounds[0,0,2])]);data.xfrc_applied[1,3:]=np.cross(arm,data.xfrc_applied[1,:3])
    mujoco.mj_step(model,data)
    for c in data.contact[:data.ncon]:pen=max(pen,-float(c.dist))
   poses.append(np.c_[data.xpos[1:],data.xquat[1:]]);vel.append(data.qvel.copy())
  return np.array(poses),pen,np.array(vel)
 idle,idle_pen,_=simulate(False);poses,pen,vel=simulate(True)
 local=[m.vertices@basis-centers[i]for i,m in enumerate(meshes)]
 camera=[]
 for pose in poses:
  camera.append([(v@Rotation.from_quat(p[[4,5,6,3]]).as_matrix().T+p[:3])@basis.T for v,p in zip(local,pose)])
 scale=max(float(np.ptp(bounds.reshape(-1,3),axis=0).max()),1e-6);rest_drift=float(np.linalg.norm(idle[:,:,:3]-idle[:1,:,:3],axis=-1).max());report.update(no_force_max_translation=rest_drift,no_force_penetration=idle_pen,active_max_penetration=pen,scene_scale=scale,static_support_actors=static,collision_geometry='native analytic flat load-bearing boxes for support graphs; convex-footprint prisms for common-surface layouts, spheres for balls; separate from depth surfaces',force_peak=peak,force_duration=2.5 if contact_target_actor is not None else 1.25,force_shape='eased finite force with bounded velocity feedback; release at requested contact',native_solver='MuJoCo implicitfast',actuation_type=actuation_type,status='candidate; not yet user accepted')
 return camera,poses,idle,local,report

def fmt(x):return ' '.join(str(float(v))for v in x)

def uniform_points(meshes,budget=1200):
 h=np.sqrt(sum(m.area for m in meshes)/budget);out=[];stats=[]
 for i,m in enumerate(meshes):
  n=max(12,round(m.area/h**2));pts,faces=trimesh.sample.sample_surface_even(m,n,radius=.62*h,seed=314159+i)
  stats.append(dict(actor=i,area=float(m.area),count=len(pts),requested=n,common_spacing=float(h),density=float(len(pts)/m.area)))
  out.append(pts)
 return out,stats

from numba import njit

@njit(cache=True)
def xpbd_pairs(x,inv,pairs,rest,compliance):
 for k in range(len(pairs)):
  i,j=pairs[k];delta=x[i]-x[j];length=np.sqrt((delta*delta).sum());denom=inv[i]+inv[j]+compliance
  if length>1e-10 and denom>0:
   change=-(length-rest[k])/denom/length
   x[i]+=inv[i]*change*delta;x[j]-=inv[j]*change*delta

def cloth_rollout(initial,faces,grip,stack_points,table_height,width):
 """Thin material mesh, localized moving grip, rest-shape bending and finite support."""
 from scipy.spatial import Delaunay
 e={}
 for a,b,c in faces:
  for i,j,k in [(a,b,c),(a,c,b),(b,c,a)]:e.setdefault(tuple(sorted((int(i),int(j)))),[]).append(int(k))
 edges=np.array(sorted(e),int);bends=np.array(sorted({tuple(sorted(v))for v in e.values()if len(v)==2 and v[0]!=v[1]}),int);length=np.linalg.norm(initial[edges[:,0]]-initial[edges[:,1]],axis=1);blength=np.linalg.norm(initial[bends[:,0]]-initial[bends[:,1]],axis=1)
 thickness=.002*width;support_height=min(float(np.quantile(stack_points[:,2],.98)),float(initial[:,2].min()-thickness));tri=Delaunay(stack_points[::max(1,len(stack_points)//2500),:2]);dt=1/144;x=initial.copy();velocity=np.zeros_like(x);frames=[x.copy()];ratios=[]
 def eased(v):v=np.clip(v,0,1);return v*v*(3-2*v)
 for step in range(1,577):
  t=step*dt
  if t<=.25:
   if step%12==0:frames.append(x.copy())
   continue
  previous=x.copy();velocity*=.985;x+=velocity*dt+np.array([0,0,-width*9.81])*dt**2;inv=np.ones(len(x));holding=t<2.65
  if holding:inv[grip]=0
  drive=eased((t-.25)/1.8);lift=np.sin(np.pi*eased((t-.25)/2.4))*.008*width;grip_x=initial[grip]+np.array([-.20*width*drive,0,lift])
  inside=tri.find_simplex(x[:,:2])>=0;floor=np.where(inside,support_height,table_height)+thickness
  for _ in range(24):
   if holding:x[grip]=grip_x
   xpbd_pairs(x,inv,edges,length,1e-9/dt**2);xpbd_pairs(x,inv,bends,blength,1e-8/dt**2)
   a=initial-initial.mean(0);b=x-x.mean(0);u,_,vt=np.linalg.svd(a.T@b);rot=u@vt
   if np.linalg.det(rot)<0:u[:,-1]*=-1;rot=u@vt
   goal=a@rot+x.mean(0);free=inv>0;x[free]+=.08*(goal[free]-x[free])
   free=inv>0;x[free,2]=np.maximum(x[free,2],floor[free])
  velocity=(x-previous)/dt;touch=x[:,2]<=floor+thickness;velocity[touch,:2]*=.80
  if step%12==0:
   frames.append(x.copy());ratios.append(np.quantile(np.linalg.norm(x[edges[:,0]]-x[edges[:,1]],axis=1)/np.maximum(length,1e-9),.999))
 report=dict(solver='stiff folded-bundle material distance/bending and soft shape-matching constraints; observed lower-stack/table support; localized grasp and release',edge_stretch_p999_max=float(max(ratios)),stack_support_height=float(support_height),table_support_height=float(table_height),fixed_global_plane_projection=False,rest_shape_preserved_at_start=True,grip_vertices=len(grip),lower_stack_is_static_prior=True)
 return np.array(frames),report

SOURCE_TEXT_AT_IMPORT=Path(__file__).read_text()
SOURCE_SHA_AT_IMPORT=hashlib.sha256(SOURCE_TEXT_AT_IMPORT.encode()).hexdigest()
