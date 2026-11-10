"""One real pinned Qwen vision block; public native primitives, no full encoder."""
import ctypes as C
import json,struct,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoProcessor
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionBlock,Qwen3VLVisionPatchEmbed,Qwen3VLVisionRotaryEmbedding,apply_rotary_pos_emb_vision
from transformers.vision_utils import get_vision_interpolation_indices_and_weights,get_vision_position_ids
sys.path.insert(0,str(Path('tests').resolve()))
from suite_numeric_api import Numeric,View,Conv,Attention
root=Path('build/metal-qualification/t06-t07-20261031')
model=Path('build/metal-oracles/models/Qwen/Qwen-Image-2.1-Turbo')
variant=sys.argv[1] if len(sys.argv)>1 else 'production'
config=Qwen3VLVisionConfig.from_dict(json.loads((model/'text_encoder/config.json').read_text())['vision_config'])
config._attn_implementation='eager'
torch.set_num_threads(2)
block=Qwen3VLVisionBlock(config).bfloat16().eval();patch=Qwen3VLVisionPatchEmbed(config).bfloat16().eval()
with safe_open(model/'text_encoder/model.safetensors',framework='pt',device='cpu') as sf:
 for module,prefix in ((block,'model.visual.blocks.0.'),(patch,'model.visual.patch_embed.')):
  module.load_state_dict({key[len(prefix):]:sf.get_tensor(key) for key in sf.keys() if key.startswith(prefix)},strict=True)
 table=sf.get_tensor('model.visual.pos_embed.weight')
processor=AutoProcessor.from_pretrained(model/'processor',local_files_only=True)
data=(root/'image-reference/fixtures/reference.ds4i').read_bytes();w,h,c=struct.unpack_from('<3I',data)
image=Image.fromarray(np.frombuffer(data,np.uint8,offset=12).reshape(h,w,c))
inputs=processor.image_processor(images=[image],return_tensors='pt')
pixels=inputs['pixel_values'].bfloat16();grid=inputs['image_grid_thw']
ix,iw=get_vision_interpolation_indices_and_weights(grid,48,mode='bilinear',align_corners=True,spatial_merge_size=2)
positions=get_vision_position_ids(grid,2)
with torch.inference_mode():
 ref=patch(pixels)
 pos=(torch.nn.functional.embedding(ix,table)*iw[:,:,None]).sum(1).bfloat16()
 x=ref+pos
 co,si=Qwen3VLVisionRotaryEmbedding(config)(x,positions)
records=[];captured={};handles=[]
def metric(name,a,b):
 a=a.reshape_as(b).to(b.dtype);d=a.float()-b.float()
 records.append(dict(stage=name,different=int((a!=b).sum()),elements=a.numel(),max_abs=float(d.abs().max())))
library=Path('tests/libds4_numeric_metal.so') if variant=='production' else root/variant/'libnumeric.so'
with Numeric('metal',library_path=library.resolve()) as n:
 weight=patch.state_dict()['proj.weight'];bias=patch.state_dict()['proj.bias']
 y=n.tensor(torch.zeros_like(ref));descriptor=Conv(pixels.shape[0],3,1152,1,(2,16,16),(2,16,16),(2,16,16),(1,1,1),(0,0,0),(0,0,0))
 n.call('conv',y,n.tensor(pixels),n.tensor(weight),n.tensor(bias),C.byref(descriptor))
 metric('patch-conv3d',n.read(y,ref.shape),ref);n.clear()
 def hook(name,m,args,out):
  if not args or not isinstance(args[0],torch.Tensor):return
  z=args[0].detach();captured[name]=(z,out.detach() if isinstance(out,torch.Tensor) else out)
  if isinstance(m,torch.nn.Linear):
   weights=m.state_dict();weight=weights['weight'];bias=weights.get('bias')
   y=n.tensor(torch.zeros_like(out));n.call('linear',y,n.tensor(z),n.tensor(weight),n.tensor(bias) if bias is not None else View(),z.numel()//z.shape[-1],weight.shape[1],weight.shape[0])
  elif isinstance(m,torch.nn.LayerNorm):
   weights=m.state_dict();y=n.tensor(torch.zeros_like(out));n.call('norm',y,n.tensor(z),n.tensor(weights['weight']),n.tensor(weights['bias']),z.shape[-1],m.eps,1,1,False)
  elif name.endswith('act_fn'):
   y=n.tensor(torch.zeros_like(out));n.call('point',y,n.tensor(z),View(),4,0)
  else:return
  metric(name,n.read(y,out.shape),out);n.clear()
 for name,m in block.named_modules():handles.append(m.register_forward_hook(lambda m,a,o,name=name:hook(name,m,a,o)))
 with torch.inference_mode():final=block(x,torch.tensor([0,x.shape[0]],dtype=torch.int32),position_embeddings=(co,si))
 for handle in handles:handle.remove()
 q,k,v=captured['attn.qkv'][1].reshape(-1,3,16,72).permute(1,0,2,3).unbind(0)
 for label,z in (('q',q),('k',k)):
  expected=apply_rotary_pos_emb_vision(z,z,co,si)[0]
  y=n.tensor(torch.zeros_like(z));n.call('rope',y,n.tensor(z),n.tensor(co[:,:36].contiguous()),n.tensor(si[:,:36].contiguous()),16,72,False,False)
  metric('rope-'+label,n.read(y,z.shape),expected);n.clear()
 q,k=apply_rotary_pos_emb_vision(q,k,co,si)
 seq=q.shape[0]
 bounds=np.tile(np.array([[0,seq]],np.uint32),(seq,1));params=Attention(1,seq,seq,16,16,72,72**-.5,0,True,True)
 y=n.tensor(torch.zeros_like(q));n.call('attend',y,n.tensor(q),n.tensor(k),n.tensor(v),bounds.ctypes.data_as(C.POINTER(C.c_uint32)),C.byref(params))
 metric('attention-same-input',n.read(y,q.shape),captured['attn.proj'][0])
(root/f'qwen-vision-{variant}-block0.json').write_text(json.dumps(records,indent=2)+'\n')
for label,tensor in (('q',q),('k',k),('v',v),('attention',captured['attn.proj'][0])):
 tensor.detach().contiguous().view(torch.uint16).numpy().tofile(root/f'qwen-vision-{label}.bf16')
print(json.dumps(records,indent=2))
