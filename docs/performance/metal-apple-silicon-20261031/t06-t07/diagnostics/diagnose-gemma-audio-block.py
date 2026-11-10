"""One pinned audio block, with independent native primitive replay."""
import ctypes as C
import json,sys,math
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioLayer,Gemma4RMSNorm
sys.path.insert(0,str(Path('tests').resolve()))
from suite_numeric_api import Numeric,View
root=Path('build/metal-qualification/t06-t07-20261031');oracle=root/'gemma-media/oracle';model=Path('build/metal-oracles/models/google/embeddinggemma-2')
config=Gemma4AudioConfig.from_dict(json.loads((model/'config.json').read_text())['audio_config'])
layer=int(sys.argv[2]) if len(sys.argv)>2 else 0
torch.set_num_threads(2);block=Gemma4AudioLayer(config,layer).bfloat16().eval();prefix=f'audio_tower.layers.{layer}.'
with safe_open(model/'model.safetensors',framework='pt',device='cpu') as sf:block.load_state_dict({k[len(prefix):]:sf.get_tensor(k) for k in sf.keys() if k.startswith(prefix)},strict=True)
def load(stage,width,index=0):return torch.from_numpy(np.fromfile(oracle/f'clip-0-{stage}-{index}.bf16',np.uint16)).view(torch.bfloat16).reshape(1,-1,width)
x=load('audio-input',1024) if layer==0 else load('audio-layer',1024,layer-1);pos=load('audio-position',1024);seq=x.shape[1];blocks=(seq+11)//12;padded=blocks*12
valid=torch.from_numpy(np.fromfile(oracle/'item-1-media-0-mask.u8',np.uint8)).bool()[::4]
mask=torch.ones(1,1,blocks,12,24,dtype=torch.bool)
for b in range(blocks):
 for r in range(12):
  q=b*12+r
  for c in range(24):
   key=b*12-12+c
   if q<seq and 0<=key<seq and bool(valid[key]) and q-11<=key<=q:mask[0,0,b,r,c]=False
variant=sys.argv[1] if len(sys.argv)>1 else 'production'
records=[];captured={};handles=[]
def metric(name,a,b):
 a=a.reshape_as(b).to(b.dtype);d=a.float()-b.float();records.append(dict(stage=name,different=int((a!=b).sum()),elements=a.numel(),max_abs=float(d.abs().max())))
library=Path('tests/libds4_numeric_metal.so') if variant=='production' else root/variant/'libnumeric.so'
with Numeric('metal',library_path=library.resolve()) as n:
 def hook(name,module,args,out):
  if not args or not isinstance(args[0],torch.Tensor):return
  z=args[0].detach();captured[name]=(z,out.detach() if isinstance(out,torch.Tensor) else out)
  if isinstance(module,torch.nn.Linear):
   w=module.state_dict()['weight'];bv=module.state_dict().get('bias');y=n.tensor(torch.zeros_like(out))
   if z.stride(-1)!=1:
    n.call('matmul_batch_layout',y,n.tensor(z.transpose(-1,-2).contiguous()),n.tensor(w),1,z.numel()//z.shape[-1],w.shape[1],w.shape[0],True,True)
   else:n.call('linear',y,n.tensor(z),n.tensor(w),n.tensor(bv) if bv is not None else View(),z.numel()//z.shape[-1],w.shape[1],w.shape[0])
  elif isinstance(module,Gemma4RMSNorm):
   w=module.state_dict().get('weight');y=n.tensor(torch.zeros_like(out))
   if z.stride(-1)!=1:n.call('channel_rms',y,n.tensor(z.transpose(-1,-2).contiguous()),n.tensor(w),z.shape[-1],z.numel()//z.shape[-1],1e-6)
   else:n.call('norm',y,n.tensor(z),n.tensor(w) if w is not None else View(),View(),z.shape[-1],1e-6,0,int(w is not None),False)
  elif isinstance(module,torch.nn.SiLU):
   y=n.tensor(torch.zeros_like(out));n.call('point',y,n.tensor(z),View(),2,0)
  else:return
  metric(name,n.read(y,out.shape),out);n.clear()
 for name,m in block.named_modules():handles.append(m.register_forward_hook(lambda m,a,o,name=name:hook(name,m,a,o)))
 with torch.inference_mode():final=block(x,mask,pos)
 for h in handles:h.remove()
 metric('upstream-block-vs-frozen',final,load('audio-layer',1024,layer))
 q=captured['self_attn.q_proj.linear'][1].float().reshape(1,seq,8,128);k=captured['self_attn.k_proj.linear'][1].float().reshape_as(q);v=captured['self_attn.v_proj.linear'][1].float().reshape_as(q)
 per=block.self_attn.state_dict()['per_dim_scale'];soft=torch.nn.functional.softplus(per)
 y=n.tensor(torch.zeros_like(per));n.call('point',y,n.tensor(per),View(),9,0);metric('scale-softplus',n.read(y,per.shape),soft);n.clear()
 def point(name,z,op,other=View(),scalar=0,expected=None):
  y=n.tensor(torch.zeros_like(z));n.call('point',y,n.tensor(z),other,op,scalar);a=n.read(y,z.shape);metric(name,a,expected);n.clear();return expected
 q=point('q-scale',q,6,scalar=block.self_attn.q_scale,expected=q*block.self_attn.q_scale)
 q=point('q-per-dim-scale',q,1,n.tensor(soft.float()),expected=q*soft)
 k=point('k-scale',k,6,scalar=block.self_attn.k_scale,expected=k*block.self_attn.k_scale)
 qp=torch.nn.functional.pad(q,(0,0,0,0,0,padded-seq)).reshape(1,blocks,12,8,128).permute(0,3,1,2,4).contiguous().reshape(8*blocks,12,128)
 kp=torch.nn.functional.pad(k,(0,0,0,0,12,11)).unfold(1,24,12).movedim(-1,2).permute(0,3,1,2,4).contiguous().reshape(8*blocks,24,128)
 vp=torch.nn.functional.pad(v,(0,0,0,0,12,11)).unfold(1,24,12).movedim(-1,2).permute(0,3,1,2,4).contiguous().reshape(8*blocks,24,128)
 def matmul(name,a,b,transpose):
  expected=a@b.transpose(-1,-2) if transpose else a@b
  rows,inner=a.shape[-2:];cols=expected.shape[-1];batch=a.shape[0]
  y=n.tensor(torch.zeros_like(expected));n.call('matmul_batch',y,n.tensor(a),n.tensor(b),batch,rows,inner,cols,transpose);actual=n.read(y,expected.shape);metric(name,actual,expected);n.clear();return expected
 ac=matmul('blocked-qk-f32',qp,kp,True).reshape(1,8,blocks,12,24)
 rel=captured['self_attn.relative_k_proj'][1].float().reshape(13,8,128).permute(1,0,2).contiguous()
 bd=matmul('relative-qk-f32',qp.reshape(8,padded,128),rel,True).reshape(1,8,blocks,12,13)
 bd=torch.nn.functional.pad(bd,(0,12)).reshape(1,8,blocks,12*25)[...,:12*24].reshape(1,8,blocks,12,24)
 capped=(ac+bd)/50
 capped=point('score-tanh',capped,10,expected=torch.tanh(capped))*50
 scores=capped.masked_fill(mask,-1e9);probs=scores.softmax(-1)
 bounds=np.zeros((8*blocks*12,2),np.uint32)
 for h in range(8):
  for b in range(blocks):
   for r in range(12):
    allowed=(~mask[0,0,b,r]).nonzero().flatten();bounds[(h*blocks+b)*12+r]=[int(allowed[0]),int(allowed[-1])+1] if allowed.numel() else [0,24]
 flat=scores.reshape(-1,24).contiguous();y=n.tensor(torch.zeros_like(flat));n.call('softmax',y,n.tensor(flat),bounds.ctypes.data_as(C.POINTER(C.c_uint32)),flat.shape[0],24,1)
 metric('probabilities-f32',n.read(y,flat.shape),probs.reshape_as(flat));n.clear()
 matmul('blocked-pv-f32',probs.reshape(8*blocks,12,24).contiguous(),vp,False)
(root/f'gemma-audio-{variant}-block{layer}-diagnosis.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))
