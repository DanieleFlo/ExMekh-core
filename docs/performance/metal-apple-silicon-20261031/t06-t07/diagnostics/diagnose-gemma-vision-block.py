"""Single pinned vision patch module/block, never the whole Gemma model."""
import ctypes as C
import json,sys
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionEncoderLayer,Gemma4VisionPatchEmbedder,Gemma4VisionRotaryEmbedding,Gemma4RMSNorm,apply_multidimensional_rope
sys.path.insert(0,str(Path('tests').resolve()))
from suite_numeric_api import Numeric,View,Attention
root=Path('build/metal-qualification/t06-t07-20261031');oracle=root/'gemma-media/oracle'
variant=sys.argv[1] if len(sys.argv)>1 else 'gemm-cpu-compatible'
model=Path('build/metal-oracles/models/google/embeddinggemma-2')
item=int(sys.argv[2]) if len(sys.argv)>2 else 0
frames=2 if item==2 else 1
first_frame=1 if item==2 else 0
config=Gemma4VisionConfig.from_dict(json.loads((model/'config.json').read_text())['vision_config'])
config._attn_implementation='eager'
torch.set_num_threads(2)
block=Gemma4VisionEncoderLayer(config,0).bfloat16().eval();patch=Gemma4VisionPatchEmbedder(config).bfloat16().eval()
with safe_open(model/'model.safetensors',framework='pt',device='cpu') as sf:
 for module,prefix in ((block,'vision_tower.encoder.layers.0.'),(patch,'vision_tower.patch_embedder.')):
  module.load_state_dict({key[len(prefix):]:sf.get_tensor(key) for key in sf.keys() if key.startswith(prefix)},strict=True)
positions=torch.from_numpy(np.fromfile(oracle/f'item-{item}-media-0-positions.i32',np.int32).astype(np.int64)).reshape(frames,-1,2)
pixels=torch.from_numpy(np.fromfile(oracle/f'item-{item}-media-0-pixels.f32',np.float32)).reshape(frames,-1,768).bfloat16()
valid=positions[...,0]>=0;seq=pixels.shape[1];records=[];captured={}
def metric(name,a,b):
 a=a.bfloat16().reshape_as(b);b=b.bfloat16();d=a.float()-b.float()
 records.append(dict(stage=name,different=int((a.view(torch.int16)!=b.view(torch.int16)).sum()),elements=a.numel(),max_abs=float(d.abs().max())))
with Numeric('metal',library_path=Path('tests/libds4_numeric_metal.so').resolve() if variant=='production' else ((root/f'{variant}/libnumeric.so') if (root/f'{variant}/libnumeric.so').exists() else (root/f'neon-model-experiment/{variant}/libnumeric.so')).resolve()) as n:
 def hook(name,module,args,output):
  if not args or not isinstance(args[0],torch.Tensor):return
  x=args[0].detach();captured[name]=(x,output.detach() if isinstance(output,torch.Tensor) else output)
  if isinstance(module,torch.nn.Linear):
   w=module.state_dict()['weight'];y=n.tensor(torch.zeros_like(output));n.call('linear',y,n.tensor(x),n.tensor(w),View(),x.numel()//x.shape[-1],w.shape[1],w.shape[0])
  elif isinstance(module,Gemma4RMSNorm):
   w=module.state_dict().get('weight');y=n.tensor(torch.zeros_like(output));n.call('norm',y,n.tensor(x),n.tensor(w) if w is not None else View(),View(),x.shape[-1],config.rms_norm_eps,0,int(w is not None),False)
  elif name.endswith('act_fn'):
   y=n.tensor(torch.zeros_like(output));n.call('point',y,n.tensor(x),View(),4,0)
  else:return
  metric(name,n.read(y,output.shape),output);n.clear()
 handles=[]
 for owner,label in ((patch,'patch.'),(block,'')):
  for name,module in owner.named_modules():handles.append(module.register_forward_hook(lambda m,a,o,name=label+name:hook(name,m,a,o)))
 with torch.inference_mode():
  x=patch(pixels,positions,~valid)
  reference=torch.from_numpy(np.concatenate([np.fromfile(oracle/f'frame-{first_frame+i}-vision-patches-0.bf16',np.uint16) for i in range(frames)])).view(torch.bfloat16).reshape_as(x)
  metric('upstream-patches-vs-frozen',x,reference)
  co,si=Gemma4VisionRotaryEmbedding(config)(x,positions)
  mask=torch.zeros(frames,1,seq,seq,dtype=torch.bfloat16).masked_fill(~valid[:,None,None,:],-float('inf'))
  final=block(reference,position_embeddings=(co,si),attention_mask=mask,position_ids=positions)
  frozen=torch.from_numpy(np.concatenate([np.fromfile(oracle/f'frame-{first_frame+i}-vision-layer-0.bf16',np.uint16) for i in range(frames)])).view(torch.bfloat16).reshape_as(final)
  metric('upstream-block-vs-frozen',final,frozen)
 for handle in handles:handle.remove()
 q=apply_multidimensional_rope(captured['self_attn.q_norm'][1],co,si,positions).contiguous()
 k=apply_multidimensional_rope(captured['self_attn.k_norm'][1],co,si,positions).contiguous();v=captured['self_attn.v_norm'][1].contiguous()
 bounds=np.array([[0,int(valid[b].sum())] for b in range(frames) for _ in range(seq)],dtype=np.uint32)
 params=Attention(frames,seq,seq,12,12,64,1,0,True,True)
 y=n.tensor(torch.zeros_like(q));n.call('attend',y,n.tensor(q),n.tensor(k),n.tensor(v),bounds.ctypes.data_as(C.POINTER(C.c_uint32)),C.byref(params))
 metric('attention-same-input',n.read(y,q.shape),captured['self_attn.o_proj.linear'][0])
(root/f'gemma-vision-{variant}-item{item}-diagnosis.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
