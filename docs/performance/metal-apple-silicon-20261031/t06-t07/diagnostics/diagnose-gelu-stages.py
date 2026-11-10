import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path('tests').resolve()))
from suite_numeric_api import Numeric, View
root=Path('build/metal-qualification/t06-t07-20261031/neon-model-experiment')
records=[]
with Numeric('metal',library_path=(root/'gemm-cpu-gelu-stages/libnumeric.so').resolve()) as n:
    for name in ('mlp.act_fn','ple_block.act_fn'):
        x=torch.from_numpy(np.fromfile(root/f'gemm-cpu-tail/{name}-input.bf16',dtype=np.uint16)).view(torch.bfloat16).float()
        inner=(2/torch.pi)**.5*(x+.044715*(x*x*x))
        expected=torch.tanh(inner)
        y=n.tensor(torch.zeros_like(x))
        n.call('point',y,n.tensor(x),View(),4,0)
        gpu_inner=n.read(y,x.shape)
        n.call('point',y,n.tensor(inner),View(),10,0)
        gpu_tanh=n.read(y,x.shape)
        def metric(a,b):
            d=(a!=b);where=d.nonzero().flatten()[:12]
            return {'different':int(d.sum()),'elements':a.numel(),'max_abs':float((a-b).abs().max()),'examples':[{'x':float(x[i]),'expected':float(b[i]),'actual':float(a[i])} for i in where]}
        records.append({'stage':name,'inner':metric(gpu_inner,inner),'tanh':metric(gpu_tanh,expected),'gelu_using_gpu_tanh':metric((.5*x*(1+gpu_tanh)).bfloat16(),(.5*x*(1+expected)).bfloat16())})
        n.clear()
(root/'gelu-stages.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
