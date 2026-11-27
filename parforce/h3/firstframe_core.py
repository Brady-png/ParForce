"""Shared first-frame data contract for training and inference."""
import torch
import torch.nn.functional as F

def assemble_firstframe(noisy_target,controls,keyframe,condition_noise,condition_t=.999):
    if keyframe.ndim!=2 or keyframe.shape!=condition_noise.shape or keyframe.shape[1]!=noisy_target.shape[1]:raise ValueError('Invalid keyframe geometry')
    if len(controls)!=len(noisy_target):raise ValueError('Control-target alignment mismatch')
    prefix=condition_t*keyframe+(1-condition_t)*condition_noise
    return torch.cat([prefix,noisy_target],dim=0),torch.cat([torch.zeros(len(keyframe),controls.shape[1],device=controls.device,dtype=controls.dtype),controls],dim=0),len(keyframe)

def generated_loss(prediction,target,condition_rows):
    generated=prediction[condition_rows:]
    if generated.shape!=target.shape:raise ValueError('Loss geometry mismatch')
    return F.mse_loss(generated.float(),target.float())
