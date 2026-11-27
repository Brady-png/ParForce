"""Read the pre-final-norm decoder feature across Transformers output API variants."""
import torch


def encode_features(encoder,input_ids,layer_index=49):
    captured={}
    def record(module,inputs,output):
        captured['features']=output[0] if isinstance(output,(tuple,list)) else output
    handle=encoder.language_model.layers[layer_index].register_forward_hook(record)
    try:
        with torch.no_grad():
            encoder(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),use_cache=False)
        features=captured['features']
        if features.ndim!=3 or features.shape[:2]!=input_ids.shape:
            raise ValueError(f'Unexpected text features {features.shape}, token input {input_ids.shape}')
        return features.detach()
    finally:
        handle.remove()
