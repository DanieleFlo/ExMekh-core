"""Bounded primitive experiment, not full-model qualification."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path('tests').resolve()))
from suite_numeric_api import Numeric

torch.set_num_threads(2)
root = Path('build/metal-qualification/t06-t07-20261031')
out = root/'simd-bf16-experiment'
out.mkdir(exist_ok=True)

def hash_tensor(value):
    data=value.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()

def compare(actual, expected):
    actual=actual.to(torch.bfloat16).contiguous()
    expected=expected.to(torch.bfloat16).contiguous()
    equal=actual.view(torch.int16)==expected.view(torch.int16)
    delta=actual.float()-expected.float()
    return {'different':int((~equal).sum()),'count':actual.numel(),
            'maximum_absolute':float(delta.abs().max()),
            'rmse':float(delta.square().mean().sqrt()),
            'sha256':hash_tensor(actual)}

def run_case(name,x,w):
    x=x.contiguous().to(torch.bfloat16);w=w.contiguous().to(torch.bfloat16)
    rows,inner=x.shape;columns=w.shape[0]
    xp=out/f'{name}-x.bf16';wp=out/f'{name}-w.bf16';yp=out/f'{name}-y.f32'
    x.view(torch.uint16).numpy().tofile(xp);w.view(torch.uint16).numpy().tofile(wp)
    command=[str(root/'probe-simd-bf16'),str(root/'probe-simd-bf16.metal'),
             str(xp),str(wp),str(yp),str(rows),str(inner),str(columns)]
    result=subprocess.run(command,text=True,capture_output=True)
    (out/f'{name}-stderr.log').write_text(result.stderr)
    if result.returncode:
        return dict(name=name,status='UNSUPPORTED' if result.returncode==5 else 'FAIL',
                    returncode=result.returncode,stderr=result.stderr)
    actual=torch.from_numpy(np.fromfile(yp,dtype=np.float32).reshape(rows,columns))
    reference=torch.nn.functional.linear(x,w)
    with Numeric('metal') as numeric:
        xv=numeric.tensor(x);wv=numeric.tensor(w)
        yv=numeric.tensor(torch.zeros(rows,columns,dtype=torch.bfloat16))
        numeric.linear(yv,xv,wv,rows,inner,columns)
        mps=numeric.read(yv,(rows,columns))
        memory=numeric.memory_usage()._asdict()
    return dict(name=name,status='MEASURED',shape=[rows,inner,columns],
                input_sha256=hash_tensor(x),weight_sha256=hash_tensor(w),
                cpu_sha256=hash_tensor(reference),
                simd_vs_cpu=compare(actual,reference),mps_vs_cpu=compare(mps,reference),
                simd_vs_mps=compare(actual,mps),mps_memory=memory,
                metadata=json.loads(result.stdout))

torch.manual_seed(20261101)
results=[]
for name,shape in [('tiny',(2,8,2)),('tail',(35,39,37)),('signed',(8,512,96))]:
    rows,inner,cols=shape
    x=torch.randn(rows,inner);w=torch.randn(cols,inner)
    if name=='tiny':x.fill_(1);w.fill_(1)
    results.append(run_case(name,x,w))
    if results[-1]['status']!='MEASURED':break
if all(r['status']=='MEASURED' for r in results):
    with safe_open('build/metal-oracles/models/google/embeddinggemma-2/model.safetensors',framework='pt',device='cpu') as sf:
        name='language_model.ple.per_layer_model_projection.weight'
        w=sf.get_tensor(name)
    x=torch.from_numpy(np.fromfile(root/'gemma-text/native-128/item-0-embeddings-0.bf16',dtype=np.uint16)).view(torch.bfloat16).reshape(-1,w.shape[1])
    results.append(run_case('gemma-ple',x,w));del x,w
    with safe_open('build/metal-models/Qwen/Qwen-Image-2.1-Turbo/text_encoder/model.safetensors',framework='pt',device='cpu') as sf:
        keys=list(sf.keys())
        qkey=next(k for k in keys if k.endswith('layers.0.self_attn.q_proj.weight') and 'language_model' in k)
        normkey=qkey.replace('self_attn.q_proj','input_layernorm')
        w=sf.get_tensor(qkey);scale=sf.get_tensor(normkey)
    x=torch.from_numpy(np.fromfile(root/'image-text/encoder-cpu-trace/text/embeddings-0.bf16',dtype=np.uint16)).view(torch.bfloat16).reshape(-1,w.shape[1])
    x=((x.float()*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+1e-6)).to(torch.bfloat16)*scale).to(torch.bfloat16)
    results.append(run_case('qwen-q0',x,w))
report={'scope':'Independent primitive experiment, unchanged production MPS and frozen gates. GPU timings overlap serial acceptance and are not benchmarks.',
        'results':results,'all_computed':all(r['status']=='MEASURED' for r in results)}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
