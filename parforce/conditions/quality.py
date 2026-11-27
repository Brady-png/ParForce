"""Fixed condition-level gates; passing is not a proof of real-world physical correctness."""
from pathlib import Path
import json,numpy as np,cv2
from scipy.spatial.transform import Rotation

POLICY=dict(min_actor_fit_iou=.85,max_rest_body_fraction=.02,max_rest_rotation_degrees=5.,max_penetration_body_fraction=.03,min_target_visible_fraction=.45,min_target_motion_pixels=4.,max_rigid_orthogonality_error=1e-6)

def check_case(row,folder):
 folder=Path(folder);j=json.loads((folder/'REPORT.json').read_text());reasons=[]
 if j.get('rigid_rotation_orthogonality_error',0)>POLICY['max_rigid_orthogonality_error']:reasons.append('rigid transform invalid')
 if min(j['firstframe_surface_iou'])<POLICY['min_actor_fit_iou']:reasons.append('first-view actor fit below fixed threshold')
 with np.load(folder/'states.npz')as s:
  p=s['positions'];poses=s['poses'];idle=s['no_force_poses'];ids=s['actor_ids'];K=s['camera_K'];basis=s['basis'];times=s['times_s'];rawz=s['depth_z']
  assert len(p)==49 and len(rawz)==49 and np.allclose(times,np.arange(49)/12)
  initialworld=p[0]@basis;body_sizes=np.array([np.ptp(initialworld[ids==i],axis=0).max()for i in range(poses.shape[1])]);drifts=np.linalg.norm(idle[:,:,:3]-idle[:1,:,:3],axis=-1).max(0)/np.maximum(body_sizes,1e-8)
  q=Rotation.from_quat(idle[..., [4,5,6,3]].reshape(-1,4));q0=Rotation.from_quat(np.tile(idle[0][:,[4,5,6,3]],(49,1)));rot=np.degrees((q*q0.inv()).magnitude()).reshape(idle.shape[:2]).max(0)
  if row.get('actuation_type')=='gravity':drifts[0]=0;rot[0]=0
  if drifts.max()>POLICY['max_rest_body_fraction']:reasons.append('no-drive position stability failed')
  if rot.max()>POLICY['max_rest_rotation_degrees']:reasons.append('no-drive rotation stability failed')
  penetration=j['physics']['active_max_penetration']/max(float(body_sizes.min()),1e-8)
  if penetration>POLICY['max_penetration_body_fraction']:reasons.append('collision penetration failed')
  target=p[:,ids==0];centers=target.mean(1);uv=(centers@K.T)[:,:2]/centers[:,2,None];move=float(np.linalg.norm(uv-uv[:1],axis=-1).max());h,w=rawz.shape[1:];pts=(target@K.T);uvp=pts[...,:2]/pts[...,2,None];visible=((uvp[...,0]>=0)&(uvp[...,0]<w)&(uvp[...,1]>=0)&(uvp[...,1]<h)&(target[...,2]>0)).mean(1)
  if move<POLICY['min_target_motion_pixels']:reasons.append('target response too small')
  if np.quantile(visible,.25)<POLICY['min_target_visible_fraction']:reasons.append('target exits view too early')
  if not(np.isfinite(p).all()and np.isfinite(rawz).all()and(rawz>0).all()):reasons.append('nonfinite state/depth')
 for name in ['particles','depth']:
  c=cv2.VideoCapture(str(folder/(name+'.mp4')));n=0;fps=c.get(5)
  while c.read()[0]:n+=1
  c.release()
  if n!=49 or abs(fps-12)>.01:reasons.append('media decode/clock mismatch')
 result=dict(passed=not reasons,reasons=reasons,policy=POLICY,rest_fraction_per_actor=drifts.tolist(),rest_rotation_per_actor=rot.tolist(),penetration_fraction=penetration,target_max_motion_pixels=move,target_visibility_p25=float(np.quantile(visible,.25)),kind='automatic technical/geometry gates; independent semantic review still required')
 (folder/'QUALITY_GATE.json').write_text(json.dumps(result,indent=2)+'\n');return result
