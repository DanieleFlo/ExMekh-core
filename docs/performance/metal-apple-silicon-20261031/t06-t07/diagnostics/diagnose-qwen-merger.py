import json,sys
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from safetensors import safe_open
sys.path.insert(0,str(Path('.').resolve()))
from tests.suite_numeric_api import Numeric,View
r=Path('build/metal-qualification/t06-t07-20261031');case=sys.argv[1] if len(sys.argv)>1 else 'reference';source=r/f'product-image-{case}-cpu-vae-mps/oracle/{case}';records=[]
torch.set_num_threads(2)
def compare(name,a,b):
 d=(a.float()-b.float());row=dict(name=name,different=int((a.float()!=b.float()).sum()),max_abs=float(d.abs().max()));records.append(row);print(row,flush=True)
with Numeric('metal') as n, safe_open('build/metal-oracles/models/Qwen/Qwen-Image-2.1-Turbo/text_encoder/model.safetensors',framework='pt') as f:
 for index,layer in [(0,8),(1,16),(2,24),(3,26)]:
  prefix=f'model.visual.deepstack_merger_list.{index}.' if index<3 else 'model.visual.merger.'
  weights={key[len(prefix):]:f.get_tensor(key) for key in f.keys() if key.startswith(prefix)}
  x=torch.from_numpy(np.fromfile(source/f'vision-{layer}.bf16',np.uint16)).view(torch.bfloat16).reshape(-1,4608 if index<3 else 1152)
  norm=F.layer_norm(x,(x.shape[-1],),weights['norm.weight'],weights['norm.bias'],1e-6);y=n.tensor(torch.zeros_like(norm));n.call('norm',y,n.tensor(x),n.tensor(weights['norm.weight']),n.tensor(weights['norm.bias']),x.shape[-1],1e-6,1,1,False);compare(f'{index}-norm',n.read(y,norm.shape),norm);n.clear();z=norm.reshape(-1,4608)
  for stage in ('linear_fc1','linear_fc2'):
   expected=F.linear(z,weights[stage+'.weight'],weights[stage+'.bias']);y=n.tensor(torch.zeros_like(expected));n.call('linear',y,n.tensor(z),n.tensor(weights[stage+'.weight']),n.tensor(weights[stage+'.bias']),z.shape[0],z.shape[1],expected.shape[1]);compare(f'{index}-{stage}',n.read(y,expected.shape),expected);n.clear();z=expected
   if stage=='linear_fc1':
    expected=F.gelu(z);y=n.tensor(torch.zeros_like(z));n.call('point',y,n.tensor(z),View(),3,0);compare(f'{index}-gelu',n.read(y,z.shape),expected);n.clear();z.view(torch.int16).numpy().tofile(r/f'qwen-merger-{index}-gelu-input.bf16');expected.view(torch.int16).numpy().tofile(r/f'qwen-merger-{index}-gelu-output.bf16');z=expected
  target=source/(f'deepstack-{index}.bf16' if index<3 else 'vision-merged.bf16')
  if target.exists():compare(f'{index}-manual-vs-frozen',z,torch.from_numpy(np.fromfile(target,np.uint16)).view(torch.bfloat16).reshape(z.shape))
(r/f'qwen-merger-{case}-diagnosis.json').write_text(json.dumps(records,indent=2)+'\n')
