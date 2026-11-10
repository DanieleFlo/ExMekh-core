"""Probe common accumulation orders and exact-dot witnesses at mismatches."""
import json
import struct
import subprocess
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)
root=Path('build/metal-qualification/t06-t07-20261031')
old=root/'simd-bf16-experiment';out=root/'dot-bf16-experiment';out.mkdir(exist_ok=True)
source=(root/'probe-dot-bf16.metal').read_text()
shapes={r['name']:r['shape'] for r in json.loads((old/'report.json').read_text())['results']}
def load(path,shape):
    return torch.from_numpy(np.fromfile(path,dtype=np.uint16)).view(torch.bfloat16).reshape(shape)
def diff(a,b):return int((a.bfloat16().view(torch.int16)!=b.bfloat16().view(torch.int16)).sum())
def round_witness(value):
    # Choose the nearest BF16 value in FP64, avoiding a second FP32 rounding.
    center=struct.unpack('<I',struct.pack('<f',value))[0]>>16
    candidates=[(struct.unpack('<f',struct.pack('<I',bits<<16))[0],bits) for bits in range(max(0,center-1),min(65535,center+1)+1)]
    return min(candidates,key=lambda item:(abs(item[0]-value),item[1]&1))[0]
records=[]
for name in ('gemma-ple','qwen-q0'):
    rows,inner,cols=shapes[name]
    xp=old/f'{name}-x.bf16';wp=old/f'{name}-w.bf16'
    x=load(xp,(rows,inner));w=load(wp,(cols,inner))
    cpu=torch.nn.functional.linear(x,w)
    cpu_float=torch.nn.functional.linear(x.float(),w.float()).bfloat16()
    simd=torch.from_numpy(np.fromfile(old/f'{name}-y.f32',dtype=np.float32).reshape(rows,cols))
    record=dict(name=name,cpu_f32_vs_cpu_bf16=diff(cpu_float,cpu),orders={},witnesses=[])
    for v,label in enumerate(('sequential','reverse','four-left','four-pair','eight-pair','kahan','torch-neon-order')):
        sp=out/f'order-{v}.metal';sp.write_text(source.replace('#define VARIANT 0',f'#define VARIANT {v}'))
        yp=out/f'{name}-order-{v}.f32'
        result=subprocess.run([str(root/'probe-simd-bf16'),str(sp),str(xp),str(wp),str(yp),str(rows),str(inner),str(cols)],text=True,capture_output=True,check=True)
        actual=torch.from_numpy(np.fromfile(yp,dtype=np.float32).reshape(rows,cols))
        record['orders'][label]=dict(vs_cpu=diff(actual,cpu),vs_simd=diff(actual,simd),metadata=json.loads(result.stdout))
    for r,c in torch.nonzero(cpu.view(torch.int16)!=simd.bfloat16().view(torch.int16)).tolist():
        witness=float((x[r].double()*w[c].double()).sum())
        correct=round_witness(witness)
        record['witnesses'].append(dict(row=r,column=c,cpu=float(cpu[r,c]),simd=float(simd[r,c].bfloat16()),simd_f32=float(simd[r,c]),
                                       dot_fp64=witness,fp64_rounded=correct))
    records.append(record)
    print(name,{k:v['vs_cpu'] for k,v in record['orders'].items()},'cpu-f32',record['cpu_f32_vs_cpu_bf16'],flush=True)
(out/'report.json').write_text(json.dumps(dict(scope='Primitive order diagnosis; neither a release gate nor a benchmark.',cases=records),indent=2)+'\n')
