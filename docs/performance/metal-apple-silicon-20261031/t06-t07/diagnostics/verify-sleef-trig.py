import json,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path('.').resolve()))
from tests.suite_numeric_api import Numeric,View
root=Path('build/metal-qualification/t06-t07-20261031')
x=torch.cat((torch.from_numpy(np.fromfile(root/'qwen-rope-expected-angles.f32',np.float32)),torch.linspace(-124,124,8192),torch.tensor([0.,-0.,1e-38,-1e-38])))
records=[]
with Numeric('metal',library_path=root/'sleef-trig/libnumeric.so') as n:
 for op,fn in ((13,torch.sin),(14,torch.cos)):
  expected=fn(x);y=n.tensor(torch.zeros_like(x));n.call('point',y,n.tensor(x),View(),op,0);actual=n.read(y,x.shape)
  record=dict(op=op,different=int((actual.view(torch.int32)!=expected.view(torch.int32)).sum()),max_abs=float((actual-expected).abs().max()),elements=x.numel());print(record,flush=True);records.append(record);n.clear()
(root/'sleef-trig/verification.json').write_text(json.dumps(records,indent=2)+'\n')
assert all(x['different']==0 for x in records)
