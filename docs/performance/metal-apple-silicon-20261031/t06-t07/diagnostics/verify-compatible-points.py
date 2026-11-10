import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path('tests').resolve()))
from suite_numeric_api import Numeric,View
root=Path('build/metal-qualification/t06-t07-20261031/neon-model-experiment')
torch.set_num_threads(2)
x=torch.arange(65536,dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
x=x[torch.isfinite(x)]
results=[]
with Numeric('metal',library_path=(root/'gemm-cpu-compatible/libnumeric.so').resolve()) as n:
 for name,op,expected in [('gelu-tanh',4,torch.nn.functional.gelu(x,approximate='tanh')),('silu',2,torch.nn.functional.silu(x))]:
  y=n.tensor(torch.zeros_like(x));n.call('point',y,n.tensor(x),View(),op,0);a=n.read(y,x.shape).bfloat16()
  different=(a!=expected)&~(torch.isnan(a)&torch.isnan(expected));indices=different.nonzero().flatten()
  results.append(dict(operation=name,different=int(different.sum()),elements=x.numel(),examples=[dict(x=float(x[i]),expected=float(expected[i]),actual=float(a[i])) for i in indices[:20]]))
  n.clear()
(root/'all-finite-bf16-points.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
