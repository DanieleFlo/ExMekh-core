"""Small same-input PLE primitives with a public alternate-library handle."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0,str(Path('tests').resolve()))
from suite_numeric_api import Numeric

torch.set_num_threads(2)
root=Path('build/metal-qualification/t06-t07-20261031')
def read(path,width):
    return torch.from_numpy(np.fromfile(path,dtype=np.uint16)).view(torch.bfloat16).reshape(-1,width)
def diff(x,y):
    a,b=x.bfloat16().reshape_as(y),y.bfloat16();d=a.float()-b.float()
    return dict(different=int((a.view(torch.int16)!=b.view(torch.int16)).sum()),elements=a.numel(),max_abs=float(d.abs().max()))
with safe_open('build/metal-oracles/models/google/embeddinggemma-2/model.safetensors',framework='pt',device='cpu') as sf:
    w=sf.get_tensor('language_model.ple.per_layer_model_projection.weight')
    scale=sf.get_tensor('language_model.ple.per_layer_projection_norm.weight')
records=[]
variant=sys.argv[1] if len(sys.argv)>1 else 'gemm-cpu-pow'
with Numeric('metal',library_path=(root/f'neon-model-experiment/{variant}/libnumeric.so').resolve()) as n:
    for item in range(3):
        x=read(root/f'gemma-text/oracle/item-{item}-embeddings-0.bf16',512)
        linear=torch.nn.functional.linear(x,w);scaled=linear*0.04419417382415922
        y=n.tensor(torch.zeros_like(linear));xv=n.tensor(x);wv=n.tensor(w)
        n.linear(y,xv,wv,len(x),512,12288)
        actual=n.read(y,linear.shape).bfloat16()
        record=dict(item=item,linear=diff(actual,linear))
        n.scale(y,y,0.04419417382415922)
        actual=n.read(y,scaled.shape).bfloat16()
        record['scale']=diff(actual,scaled)
        z=scaled.reshape(-1,512).float()
        expected=(z*(z.square().mean(-1,keepdim=True)+1e-6).pow(-0.5)*scale.float()).bfloat16().reshape_as(scaled)
        n.rms_norm(y,y,n.tensor(scale),512,1e-6)
        actual=n.read(y,scaled.shape).bfloat16()
        record['norm']=diff(actual,expected)
        record['cpu_vs_trace']=diff(expected,read(root/f'gemma-text/oracle/item-{item}-ple-0.bf16',12288))
        records.append(record);n.clear()
(root/f'neon-model-experiment/{variant}/ple-diagnosis.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
