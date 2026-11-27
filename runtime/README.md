# Inference runtime snapshot

This is the reviewed VideoX-Fun runtime used for H3 Teacher and Student. It includes the Ascend empty audio index guard and the ParForce control adapter path. The snapshot is outside the GitHub Pages repository. The associated VideoX-Fun notice is in `licenses/`.

Test environment: Python 3.12, PyTorch 2.7.1, matching torch_npu, CANN 9.0.1, Diffusers 0.40.0, Transformers 5.14.1, PEFT 0.20.0, safetensors 0.8.0, tokenizers 0.22.2. An existing compatible Ascend installation can pass its Python overlay with `--dependencies`.
