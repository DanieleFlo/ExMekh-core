"""Small public-API checks of the alternate libraries, not a model gate."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path('tests').resolve()))
from suite_numeric_api import Numeric, View

torch.set_num_threads(2)
torch.manual_seed(20261101)
root=Path('build/metal-qualification/t06-t07-20261031')
experiment=root/'neon-model-experiment'
records=[]
for variant in (sys.argv[1:] or ('gemm','gemm-rms','gemm-cpu-pow')):
    checks=[]
    with Numeric('metal',library_path=(experiment/variant/'libnumeric.so').resolve()) as numeric:
        for rows,inner,cols in [(2,8,2),(35,39,37),(3,7,5),(8,512,96),(3,33,8)]:
            x=torch.randn(rows,inner).bfloat16();w=torch.randn(cols,inner).bfloat16();b=torch.randn(cols).bfloat16()
            for bias in (None,b):
                xv=numeric.tensor(x);wv=numeric.tensor(w)
                bv=numeric.tensor(bias) if bias is not None else View()
                yv=numeric.tensor(torch.zeros(rows,cols).bfloat16())
                before=numeric.memory_usage().live_bytes
                numeric.call('linear',yv,xv,wv,bv,rows,inner,cols)
                actual=numeric.read(yv,(rows,cols)).bfloat16()
                expected=torch.nn.functional.linear(x,w,bias)
                different=int((actual.view(torch.int16)!=expected.view(torch.int16)).sum())
                assert different==0,(variant,rows,inner,cols,different)
                assert numeric.memory_usage().live_bytes==before
                checks.append(dict(kind='linear',shape=[rows,inner,cols],bias=bias is not None,bit_exact=True))
                numeric.clear()
        if variant in ('gemm-cpu-pow','gemm-cpu-precise-rsqrt','gemm-cpu-norm-precise','gemm-cpu-compatible'):
            rms=root/'rms-bf16-experiment'
            for case in json.loads((rms/'report.json').read_text())['cases']:
                name=case['name'];shape=case['shape'];pre=name.startswith('qwen')
                x=torch.from_numpy(np.fromfile(rms/f'{name}-x.bf16',dtype=np.uint16)).view(torch.bfloat16).reshape(shape)
                w=torch.from_numpy(np.fromfile(rms/f'{name}-w.bf16',dtype=np.uint16)).view(torch.bfloat16)
                variance=x.float().pow(2).mean(-1,keepdim=True)+1e-6
                normalized=x.float()*(torch.rsqrt(variance) if pre else torch.pow(variance,-0.5))
                expected=(normalized.bfloat16()*w if pre else normalized*w.float()).bfloat16()
                for in_place in (False,True):
                    xv=numeric.tensor(x);wv=numeric.tensor(w)
                    yv=xv if in_place else numeric.tensor(torch.zeros_like(x))
                    numeric.call('norm',yv,xv,wv,View(),w.numel(),1e-6,0,1,pre)
                    actual=numeric.read(yv,shape).bfloat16()
                    different=int((actual.view(torch.int16)!=expected.view(torch.int16)).sum())
                    assert different==0,(name,in_place,different)
                    checks.append(dict(kind='rms',name=name,in_place=in_place,bit_exact=True))
                    numeric.clear()
        assert numeric.memory_usage().live_bytes==0
    records.append(dict(variant=variant,status='PASS',checks=checks))
(experiment/('primitive-verification-'+sys.argv[1]+'.json' if len(sys.argv)>1 else 'primitive-verification.json')).write_text(json.dumps(dict(scope='Bounded primitive checks with alternate libraries and public Numeric API; no full-model qualification.',variants=records),indent=2)+'\n')
print('PASS',sum(len(r['checks']) for r in records),'primitive checks, including bias, tails, in-place RMS and zero live storage after cleanup')
