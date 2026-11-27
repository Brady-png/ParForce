"""Additional rank16 QKV residual at External Force token slots only."""
import torch
from torch import nn

class ForceRoutingState:
    force_mask=None

class ForceRoutedLinear(nn.Module):
    def __init__(self,shared_layer,rank,state):
        super().__init__();self.shared_layer=shared_layer;self.state=state
        self.force_lora_A=nn.Linear(shared_layer.in_features,rank,bias=False)
        self.force_lora_B=nn.Linear(rank,shared_layer.out_features,bias=False)
    def forward(self,x):
        result=self.shared_layer(x);mask=self.state.force_mask
        if mask is None or mask.ndim!=1 or len(mask)!=x.shape[1]:raise ValueError('Invalid External Force token mask')
        indices=mask.nonzero(as_tuple=True)[0]
        if not len(indices):raise ValueError('Empty External Force segment, including header')
        force=self.force_lora_B(self.force_lora_A(x.index_select(1,indices).to(self.force_lora_A.weight.dtype)))
        return result.index_add(1,indices,force.to(result.dtype))

def install_force(model,rank):
    state=ForceRoutingState()
    for block in model.control_blocks:
        for name in ('to_q','to_k','to_v'):
            setattr(block.attn,name,ForceRoutedLinear(getattr(block.attn,name),rank,state))
    model.force_routing=state

def route_name(name):
    if name.startswith('control_blocks.'):
        for projection in ('to_q','to_k','to_v'):
            marker='.attn.'+projection+'.'
            if marker in name:return name.replace(marker,marker+'shared_layer.',1)
    return name
