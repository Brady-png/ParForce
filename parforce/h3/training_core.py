"""Pure tensor contracts used by the physical H3 trainer."""
import torch


def accumulation_for(global_batch, world_size):
    if global_batch % world_size:
        raise ValueError('global_batch must divide exactly by world_size at microbatch 1')
    return global_batch // world_size


def patchify(latents, patch):
    b,c,t,h,w=latents.shape
    pt,ph,pw=patch
    if b!=1 or t%pt or h%ph or w%pw:
        raise ValueError(f'Invalid latent shape {latents.shape} for patch {patch}')
    return latents.reshape(b,c,t//pt,pt,h//ph,ph,w//pw,pw).permute(0,2,4,6,1,3,5,7).reshape(-1,c*pt*ph*pw)


def control_rows(particle, depth, has_particle, patch):
    if particle.shape!=depth.shape:
        raise ValueError('Control modalities must share latent geometry')
    p=patchify(particle,patch) if has_particle else torch.zeros_like(patchify(depth,patch))
    return torch.cat([p,patchify(depth,patch)],dim=-1)


def flow_pair(x0, noise, t):
    return t*x0+(1-t)*noise,x0-noise
