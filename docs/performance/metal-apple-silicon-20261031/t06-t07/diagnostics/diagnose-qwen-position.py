import ctypes as C,json,sys
from pathlib import Path
import numpy as np,torch
from safetensors import safe_open
root=Path('build/metal-qualification/t06-t07-20261031')
prepared=Path('build/metal-oracles').resolve();sys.path.insert(0,str(prepared/'deps'));sys.path.insert(0,str(next((prepared/'sources').glob('transformers-*'))/'src'))
from transformers.vision_utils import get_vision_interpolation_indices_and_weights
oracle=root/'product-image-reference-cpu-vae-mps/oracle/reference';native=root/'product-encoder-reference-final/native'
inputs=np.load(oracle/'inputs.npz');indices,weights=get_vision_interpolation_indices_and_weights(torch.from_numpy(inputs['image_grid_thw']),48,mode='bilinear',align_corners=True,spatial_merge_size=2)
with safe_open(str(prepared/'models/Qwen/Qwen-Image-2.1-Turbo/text_encoder/model.safetensors'),framework='pt') as f:table=f.get_tensor('model.visual.pos_embed.weight')
values=table[indices].float();expected=(values*weights[:,:,None]).sum(1)
values=np.ascontiguousarray(values.permute(0,2,1).numpy());weights=np.ascontiguousarray(weights[:,None,:].expand(-1,1152,-1).numpy());size=expected.numel()
lib=C.CDLL(str((root/'probe-qwen-position.so').resolve()));ptr=C.POINTER(C.c_float)
patch=torch.from_numpy(np.fromfile(native/'vision_patch-0.bf16',np.uint16)).view(torch.bfloat16).reshape(-1,1152)
actual=torch.from_numpy(np.fromfile(native/'vision_input-0.bf16',np.uint16)).view(torch.bfloat16).reshape(-1,1152)
results={}
for name in ('position','unfused_position'):
 fn=getattr(lib,name);fn.argtypes=[ptr,ptr,ptr,C.c_size_t];out=np.empty(size,np.float32);fn(values.ctypes.data_as(ptr),weights.ctypes.data_as(ptr),out.ctypes.data_as(ptr),size)
 out=torch.from_numpy(out).reshape(expected.shape);z=patch+out.bfloat16()
 results[name]=dict(position_f32_diff=int((out!=expected).sum()),position_bf16_diff=int((out.bfloat16()!=expected.bfloat16()).sum()),vision_input_cpu_diff=int((z!=(patch+expected.bfloat16())).sum()),matches_actual_diff=int((z!=actual).sum()))
angles=np.empty(16*18,np.float32);co=np.empty_like(angles);si=np.empty_like(angles);lib.frequency.argtypes=[ptr,ptr,ptr];lib.frequency(*[x.ctypes.data_as(ptr) for x in (angles,co,si)])
expected_angles=torch.arange(16).float()[:,None]*(1/(10000**(torch.arange(0,36,2).float()/36)))
results['frequency']=dict(angles_diff=int((torch.from_numpy(angles).reshape(16,18)!=expected_angles).sum()),libm_cos_diff=int((torch.from_numpy(co).reshape(16,18)!=expected_angles.cos()).sum()),libm_sin_diff=int((torch.from_numpy(si).reshape(16,18)!=expected_angles.sin()).sum()))
print(json.dumps(results,indent=2));(root/'qwen-position-diagnosis.json').write_text(json.dumps(results,indent=2)+'\n')
expected_angles.numpy().tofile(root/'qwen-rope-expected-angles.f32');angles.tofile(root/'qwen-rope-host-angles.f32')
