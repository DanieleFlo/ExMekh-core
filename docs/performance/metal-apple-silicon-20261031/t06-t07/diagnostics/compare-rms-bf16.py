"""Compare RMS arithmetic variants on identical captured activations."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path('tests').resolve()))
from suite_numeric_api import Numeric, View

torch.set_num_threads(2)
root=Path('build/metal-qualification/t06-t07-20261031')
out=root/'rms-bf16-experiment';out.mkdir(exist_ok=True)
shader=(root/'probe-rms-bf16.metal').read_text()

def load(path,width):
    return torch.from_numpy(np.fromfile(path,dtype=np.uint16)).view(torch.bfloat16).reshape(-1,width)

def metrics(actual,expected):
    actual=actual.bfloat16();expected=expected.bfloat16()
    delta=actual.float()-expected.float()
    return dict(different=int((actual.view(torch.int16)!=expected.view(torch.int16)).sum()),
                elements=actual.numel(),max_abs=float(delta.abs().max()),
                rmse=float(delta.square().mean().sqrt()))

cases=[]
with safe_open('build/metal-oracles/models/google/embeddinggemma-2/model.safetensors',framework='pt',device='cpu') as sf:
    for layer in (0,6,18,23):
        path=root/'gemma-text/oracle'/('item-0-embeddings-0.bf16' if not layer else f'item-0-layer-{layer-1}.bf16')
        x=load(path,512);w=sf.get_tensor(f'language_model.layers.{layer}.input_layernorm.weight')
        cases.append((f'gemma-{layer}',x,w,False))
with safe_open('build/metal-models/Qwen/Qwen-Image-2.1-Turbo/text_encoder/model.safetensors',framework='pt',device='cpu') as sf:
    prefix=next(k for k in sf.keys() if k.endswith('layers.0.self_attn.q_proj.weight') and 'language_model' in k).split('layers.0.')[0]
    for layer in (0,6,18,35):
        path=root/'image-text/encoder-cpu-trace/text'/('embeddings-0.bf16' if not layer else f'text-{layer-1}.bf16')
        x=load(path,4096);w=sf.get_tensor(f'{prefix}layers.{layer}.input_layernorm.weight')
        cases.append((f'qwen-{layer}',x,w,True))
        if layer in (0,6):
            norm=(x.float()*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+1e-6)).bfloat16()*w
            q=torch.nn.functional.linear(norm,sf.get_tensor(f'{prefix}layers.{layer}.self_attn.q_proj.weight'))
            qw=sf.get_tensor(f'{prefix}layers.{layer}.self_attn.q_norm.weight')
            cases.append((f'qwen-qnorm-{layer}',q.reshape(-1,qw.numel()),qw,True))
reports=[]
with Numeric('metal') as native:
    for name,x,w,pre in cases:
        value=x.float();v=value.pow(2).mean(-1,keepdim=True)+1e-6
        reference=value*(torch.rsqrt(v) if pre else torch.pow(v,-0.5))
        reference=((reference.bfloat16()*w) if pre else reference*w.float()).bfloat16()
        xv=native.tensor(x);wv=native.tensor(w);yv=native.tensor(torch.zeros_like(x))
        native.call('norm',yv,xv,wv,View(),w.numel(),1e-6,0,1,pre)
        production=native.read(yv,x.shape)
        xp=out/f'{name}-x.bf16';wp=out/f'{name}-w.bf16';yp=out/f'{name}-y.f32'
        x.view(torch.uint16).numpy().tofile(xp);w.view(torch.uint16).numpy().tofile(wp)
        record=dict(name=name,shape=list(x.shape),production=metrics(production,reference),variants={})
        for variant,label in enumerate(('divide-sqrt','multiply-reciprocal','multiply-rsqrt','multiply-pow','cpu-mean-reciprocal','cpu-mean-pow')):
            sp=out/f'variant-{variant}-{int(pre)}.metal'
            sp.write_text(shader.replace('#define VARIANT 0',f'#define VARIANT {variant}').replace('#define ROUND_NORMALIZED 1',f'#define ROUND_NORMALIZED {int(pre)}'))
            command=[str(root/'probe-simd-bf16'),str(sp),str(xp),str(wp),str(yp),str(x.shape[0]),str(w.numel()),str(w.numel()),'norm_bf16']
            result=subprocess.run(command,text=True,capture_output=True,check=True)
            actual=torch.from_numpy(np.fromfile(yp,dtype=np.float32).reshape(x.shape))
            record['variants'][label]=dict(vs_cpu=metrics(actual,reference),vs_production=metrics(actual,production))
        reports.append(record);native.clear()
report=dict(scope='Same-input primitive diagnosis; no production change or gate promotion.',cases=reports,
            sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'probe-simd-bf16.m',root/'probe-rms-bf16.metal',Path(__file__))})
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
for r in reports:print(r['name'],'production',r['production']['different'],{k:v['vs_cpu']['different'] for k,v in r['variants'].items()},flush=True)
