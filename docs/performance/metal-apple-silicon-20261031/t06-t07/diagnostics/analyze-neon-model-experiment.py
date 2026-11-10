"""Read frozen traces only; never instantiate an oracle or native model."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path('.').resolve()))
from tools.qualify_metal_suite import compare_boundaries, compare_vectors
from tools.qualify_qwen_image_pipeline import compare_boundary

def bf16(path):
    values=np.fromfile(path,dtype=np.uint16)
    if not values.size or path.stat().st_size!=values.size*2:
        raise ValueError(f'empty or truncated BF16 tensor: {path}')
    return (values.astype(np.uint32)<<16).view(np.float32)

root=Path('build/metal-qualification/t06-t07-20261031')
experiment=root/'neon-model-experiment'
limits=json.loads(Path('docs/performance/native-suite-t00-20261020/tolerances.json').read_text())['qwen_latent_boundary']
variants=[]
for name in ('gemm','gemm-rms','gemm-cpu-pow','gemm-cpu-attention','gemm-cpu-precise','gemm-cpu-precise-rsqrt','gemm-cpu-norm-precise','gemm-cpu-tail','gemm-cpu-compatible'):
    folder=experiment/name
    record=dict(variant=name,scope='Experimental binary; production numeric code unchanged',gemma={'status':'NOT_VERIFIED'},encoder={'status':'NOT_VERIFIED'})
    for kind in ('gemma','encoder'):
        report=folder/f'run-{kind}/report.json'
        if report.exists():record[kind]['execution']=json.loads(report.read_text())
    if (folder/'gemma/vectors.f32').exists():
        hidden=compare_boundaries(folder/'gemma',root/'gemma-text/oracle')
        vector=compare_vectors(folder/'gemma/vectors.f32',root/'gemma-text/oracle/vectors.npy',128)
        record['gemma'].update(internal_boundaries=hidden,output_vectors=vector,status='PASS' if hidden['status']==vector['status']=='PASS' else 'FAIL')
    candidate=folder/'encoder/conditioning-0.bf16'
    if candidate.exists():
        actual=bf16(candidate)
        expected=bf16(root/'image-text/encoder-cpu-trace/text/conditioning.bf16')
        boundaries=[]
        for stage in ('embeddings-0',*[f'text-{i}' for i in range(36)]):
            reference=root/f'image-text/encoder-cpu-trace/text/{stage}.bf16'
            native=folder/f'encoder/{stage}.bf16'
            if native.exists():
                a,b=bf16(native),bf16(reference)
                metrics=compare_boundary(b,a,limits)
                boundaries.append(dict(stage=stage,**metrics))
        metrics=compare_boundary(expected,actual,limits)
        source=root/'image-text/encoder-cpu-trace/text'
        identity=dict(tokens=(source/'tokens.u32').read_bytes()==(folder/'encoder/tokens.u32').read_bytes(),
            positions=(source/'positions.u32').read_bytes()==(folder/'encoder/positions.u32').read_bytes(),
            image_mask=np.array_equal(np.fromfile(source/'tokens.u32',dtype=np.uint32)==151655,np.fromfile(folder/'encoder/image_mask.u8',dtype=np.uint8)),
            complete_layers=len(boundaries)==37)
        passed=all(identity.values()) and metrics['verdict']=='PASS' and all(b['verdict']=='PASS' for b in boundaries)
        record['encoder'].update(status='PASS' if passed else 'FAIL',input_identity=identity,conditioning=metrics,layers=boundaries)
    variants.append(record)
report=dict(scope='Saved local full-model experimental traces; inference success and numerical acceptance are independent. No production qualification.',variants=variants)
(experiment/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
for r in variants:
    print(r['variant'],'Gemma',r['gemma']['status'],'conditioning',r['encoder']['status'])
