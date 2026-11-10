"""FP32 diagnostics of BF16 Welford; public F32 views carry exact BF16 values."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
sys.path.insert(0,str(Path('tests').resolve()))
from suite_numeric_api import Numeric,View
root=Path('build/metal-qualification/t06-t07-20261031/neon-model-experiment')
variant=sys.argv[1] if len(sys.argv)>1 else 'gemm-cpu-layernorm-stats'
records=[]
with Numeric('metal',library_path=(root/f'{variant}/libnumeric.so').resolve()) as n:
 for layer,width in ((0,128),(1,32)):
  name=f'layer{layer}.norm'
  x=torch.from_numpy(np.fromfile(root/f'gemm-cpu-layernorm/audio-stem-{name}-input.bf16',np.uint16)).view(torch.bfloat16).reshape(-1,width)
  with safe_open('build/metal-oracles/models/google/embeddinggemma-2/model.safetensors',framework='pt',device='cpu') as sf:w=sf.get_tensor(f'audio_tower.subsample_conv_projection.layer{layer}.norm.weight')
  expected,mean,inv=torch.native_layer_norm(x,(width,),w.float(),None,1e-6)
  y=n.tensor(torch.zeros_like(x,dtype=torch.float32))
  n.call('norm',y,n.tensor(x.float()),n.tensor(w.float()),View(),width,1e-6,1,1,False)
  stats=n.read(y,x.shape)
  def metrics(a,b):
   return dict(different=int((a!=b).sum()),max_abs=float((a-b).abs().max()))
  shift=-inv*mean
  manual=((x.float()*inv+shift)*w.float()).bfloat16()
  fused=((x.double()*inv.double()+shift.double()).float()*w.float()).bfloat16()
  records.append(dict(stage=name,mean=metrics(stats[:,0],mean.flatten()),rstd=metrics(stats[:,2],inv.flatten()),affine_unfused=metrics(manual,expected),affine_fused=metrics(fused,expected)))
  n.clear()
(root/f'{variant}-statistics.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))
