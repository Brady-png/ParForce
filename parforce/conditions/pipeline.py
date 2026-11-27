from pathlib import Path
import sys,os,json,time,argparse,subprocess,hashlib
import cv2,numpy as np,trimesh
from scipy.spatial.transform import Rotation
from .firstview_v2 import *
from .raster import raster_accelerated
from .rigid_particles import PROFILE, sample_rigid_surfaces, fixed_marker_image
FF=os.environ.get('PARFORCE_FFMPEG','ffmpeg')

def video(path,frames):
 h,w=frames.shape[1:3];subprocess.run([str(FF),'-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r','12','-i','pipe:0','-an','-c:v','libx264','-threads','1','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],input=np.ascontiguousarray(frames).tobytes(),check=True)

def run(row,output_root):
 start=time.monotonic();out=Path(output_root)/'cases'/row['case'];out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
 if row['material']!='rigid':raise NotImplementedError('cloth route is a separate material solver')
 image=cv2.imread(row['image']);assert digest(row['image'])==row['first_frame_sha256']
 with np.load(row['depth_file'])as z:depth=clean_depth(z['depth']);K=z['K'].astype(float)
 mr=json.loads((Path(row['mask_dir'])/'RECEIPT.json').read_text());assert mr['first_frame_sha256']==row['first_frame_sha256'];h,w=depth.shape;image=cv2.resize(image,(w,h));masks=np.array([cv2.resize(cv2.imread(str(Path(row['mask_dir'])/f"actor_{row.get('mask_actor_ids',list(range(len(row['actors']))))[i]:02d}.png"),0),(w,h),interpolation=cv2.INTER_NEAREST)>0 for i in range(len(row['actors']))]);
 for index,actor_id in enumerate(row.get('mask_actor_ids',list(range(len(row['actors']))))):
  saved=next((a for a in mr['actors']if a.get('id')==actor_id),{})
  if saved.get('mask_sha256'):assert digest(Path(row['mask_dir'])/f'actor_{actor_id:02d}.png')==saved['mask_sha256']
 masks_before=masks.copy();masks=disjoint_masks(masks);cloud=backproject(depth,K);up,up_report=infer_up(cloud,masks,row['actors']);basis=basis_from_up(up)
 meshes=[]
 for mask,actor in zip(masks,row['actors']):
  primitive=actor.get('primitive');fraction=.8 if primitive in ['bottle','tapered_cup','pointed_prism','cup']else .4
  mesh=observed_sphere(depth,K,mask)if primitive=='sphere'else observed_mesh(depth,K,mask,step=1 if mask.sum()<1500 else 2,closed=True,thickness_fraction=fraction)
  meshes.append(mesh)
 shared_back=complete_shared_hidden_support(meshes,row['actors'])
 root_calibration=calibrate_root_depths(meshes,row['actors'],basis)
 bg=background_depth(depth,masks);ratios=[]
 for m,mask in zip(meshes,masks):
  dz=raster_accelerated(m.vertices,m.faces,K,(h,w));valid=mask&(dz>0)
  if valid.any():ratios.append(float(np.quantile(bg[valid]/dz[valid],.02)))
 group_visibility_scale=min(1.,min(ratios)*.985)
 if group_visibility_scale<.65:raise ValueError('Foreground/background depth ordering too inconsistent; revisit perception')
 for m in meshes:m.vertices*=group_visibility_scale
 camera,poses,idle,local,physics=rigid_rollout(meshes,row['actors'],basis,out,axis=row['axis'],force_gain=row['force_gain'],contact_target_actor=row.get('contact_target_actor'),direction_camera=row.get('direction_camera'),actuation_type=row.get('actuation_type','push'),force_height_fraction=row.get('force_height_fraction',0.),motion_scale=row.get('motion_scale',1.))
 surfaces=[];actor_visible=[];bg=background_depth(depth,masks)
 for cam in camera:
  z=bg.copy();owner=np.full((h,w),-1,np.int16)
  for i,(v,m)in enumerate(zip(cam,meshes)):
   dz=raster_accelerated(v,m.faces,K,(h,w));visible=(dz>0)&(dz<z);z[visible]=dz[visible];owner[visible]=i
  surfaces.append(z);actor_visible.append(owner)
 points,counts=sample_rigid_surfaces(meshes,np.count_nonzero(actor_visible[0]>=0),w)
 centers=np.array([(v.min(0)+v.max(0))/2 for v in [m.vertices@basis for m in meshes]])
 local_points=[point@basis-centers[i] for i,point in enumerate(points)]
 particles=np.array([np.concatenate([(lp@Rotation.from_quat(p[[4,5,6,3]]).as_matrix().T+p[:3])@basis.T for p,lp in zip(pose,local_points)]) for pose in poses]);surfaces=np.array(surfaces,np.float32);owners=np.array(actor_visible);near,far=np.quantile(particles[...,2],[.01,.99]);inv=1/surfaces;lo,hi=np.quantile(inv,[.02,.98]);dframes=np.repeat(np.uint8(255*np.clip((inv-lo)/max(hi-lo,1e-6),0,1))[...,None],3,-1);pframes=np.array([fixed_marker_image(p,K,(h,w),near,far)for p in particles]);video(out/'depth.mp4',dframes);video(out/'particles.mp4',pframes)
 ious=[]
 for i,mask in enumerate(masks):
  pred=owners[0]==i;ious.append(float((pred&mask).sum()/max(1,(pred|mask).sum())))
 np.savez_compressed(out/'states.npz',positions=particles,poses=poses,no_force_poses=idle,depth_z=surfaces,background_depth=bg,camera_K=K,basis=basis,origin=np.zeros(3),times_s=TIMES,actor_ids=np.repeat(np.arange(len(points)),list(map(len,points))),initial_geometry_masks=owners[0],render_owner=owners)
 for i,m in enumerate(meshes):m.export(out/f'actor_{i:02d}_initial_camera.ply')
 cv2.imwrite(str(out/'FIRST_FRAME.png'),image);cv2.imwrite(str(out/'DEPTH_INITIAL.png'),dframes[0]);cv2.imwrite(str(out/'PARTICLES_INITIAL.png'),pframes[0]);overlay=image.copy()
 for i,mask in enumerate(masks):
  predicted=(owners[0]==i);cont,_=cv2.findContours(predicted.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(overlay,cont,-1,(30,255,30),1)
 cv2.imwrite(str(out/'INITIAL_OVERLAY.png'),overlay)
 rotation=Rotation.from_quat(poses[..., [4,5,6,3]].reshape(-1,4)).as_matrix();orth=float(np.max(np.abs(rotation.transpose(0,2,1)@rotation-np.eye(3))));scale=float(physics['scene_scale']);report=dict(case=row['case'],status='candidate_for_user_review',firstframe_surface_iou=ious,gravity=up_report,physics=physics,root_depth_calibration=root_calibration,visibility_group_depth_scale=group_visibility_scale,mask_overlap_pixels_before=int((masks_before.sum(0)>1).sum()),mask_ownership='maximum interior distance in overlapping SAM proposals; diagnostic fit is not independent ground truth',particle_sampling=counts,particle_total=particles.shape[1],sparse_particle_profile=PROFILE,shape_model=[dict(primitive=a.get('primitive'),completion=m.metadata)for a,m in zip(row['actors'],meshes)],rigid_rotation_orthogonality_error=orth,all_finite=bool(np.isfinite(surfaces).all()and np.isfinite(particles).all()),no_force_translation_fraction=physics['no_force_max_translation']/scale,raw_future_rgb_used=False,old_trajectories_used=False,input_sha256=dict(first_frame=digest(row['image']),depth=digest(row['depth_file']),mask_receipt=digest(Path(row['mask_dir'])/'RECEIPT.json')),seconds=time.monotonic()-start,source_code_sha256=SOURCE_SHA_AT_IMPORT,user_accepted=False)
 (out/'source_used.py').write_text(SOURCE_TEXT_AT_IMPORT)
 (out/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print('DONE',row['case'],'iou',ious,'rest',report['no_force_translation_fraction'],'particles',particles.shape[1],flush=True)
 return report
