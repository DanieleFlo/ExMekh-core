import json,sys
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
sys.path.insert(0,str(Path('.').resolve()))
from tests.suite_numeric_api import Numeric,View
root=Path('build/metal-qualification/t06-t07-20261031');torch.set_num_threads(2)
x=torch.arange(65536,dtype=torch.int32).to(torch.uint16).view(torch.bfloat16);x=x[torch.isfinite(x)];torch.manual_seed(20261101)
cases=[('all-finite',x),('finite-shuffled',x[torch.randperm(x.numel())]),*[(f'merger-{i}',torch.from_numpy(np.fromfile(root/f'qwen-merger-{i}-gelu-input.bf16',np.uint16)).view(torch.bfloat16)) for i in range(4)]]
results=[]
with Numeric('metal',library_path=root/'neon-erf/libnumeric.so') as n:
 for name,x in cases:
  expected=F.gelu(x);y=n.tensor(torch.zeros_like(x));n.call('point',y,n.tensor(x),View(),3,0);actual=n.read(y,x.shape).bfloat16();different=actual.view(torch.int16)!=expected.view(torch.int16);indices=different.nonzero().flatten();row=dict(case=name,different=int(different.sum()),elements=x.numel(),examples=[dict(x=float(x[i]),actual=float(actual[i]),expected=float(expected[i])) for i in indices[:12]]);print(row,flush=True);results.append(row);n.clear()
(root/'neon-erf/verification.json').write_text(json.dumps(results,indent=2)+'\n')
assert all(row['different']==0 for row in results)
