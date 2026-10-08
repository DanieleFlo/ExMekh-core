#!/usr/bin/env python3
"""Compare dense weight streaming with a matched resident or KV-offload path."""
import argparse
import json
from pathlib import Path
import re
from perf_qwen_flash_strata import ROOT, run_bench, compare_logits, write_json
from qualify_qwen27_speculative import summarize_cycles

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,default=ROOT/'ds4-bench')
    p.add_argument('--baseline',type=Path)
    p.add_argument('--reuse-baseline',type=Path,
        help='reuse one completed matched diagnostic run directory; only with --samples 1')
    p.add_argument('--baseline-mode',choices=['resident','kv-offload'],default='resident')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,default=ROOT/'gguf/Qwen3.8-27B-UD-Q4_K_S.gguf')
    p.add_argument('--prompt',type=Path,default=ROOT/'speed-bench/promessi_sposi.txt')
    p.add_argument('--frontier',type=int,default=256)
    p.add_argument('--context',type=int,default=2048)
    p.add_argument('--chunk',type=int,default=256)
    p.add_argument('--tokens',type=int,default=32)
    p.add_argument('--resident-mb',type=int,default=0)
    p.add_argument('--mtp',action='store_true')
    p.add_argument('--warmup',type=int,default=0)
    p.add_argument('--decode-warmup',type=int,default=0)
    p.add_argument('--samples',type=int,default=1)
    p.add_argument('--timeout',type=int,default=3600)
    a=p.parse_args()
    if not 0<a.frontier or a.frontier+a.tokens>=a.context or a.context>65536:
        p.error('leave the benchmark one cell: frontier + generated tokens < context <=65536')
    if a.reuse_baseline and a.samples!=1:
        p.error('a reused single diagnostic is not five alternating A/B pairs')
    if a.reuse_baseline:
        reference=json.loads((a.reuse_baseline/'run.json').read_text(encoding='utf-8-sig'))
        if reference.get('status')!='COMPLETE' or reference.get('warmup')!=a.warmup:
            p.error('reused baseline must be complete with matching warm-up')
        cmd=reference['command']
        expected={'--ctx-start':str(a.frontier),'--ctx-max':str(a.frontier),
            '--ctx-alloc':str(a.context),'--gen-tokens':str(a.tokens),'--prefill-chunk':str(a.chunk),
            '-m':str(a.model.resolve()),'--prompt-file':str(a.prompt.resolve())}
        for flag,value in expected.items():
            if flag not in cmd or cmd[cmd.index(flag)+1]!=value:
                p.error(f'reused baseline differs at {flag}')
        if ('--mtp' in cmd)!=a.mtp or (a.mtp and cmd[cmd.index('--mtp-draft')+1]!='5'):
            p.error('reused baseline differs at NextN5')
        refenv=reference['environment']
        if refenv.get('DS4_BENCH_DECODE_WARMUP','0')!=str(a.decode_warmup):
            p.error('reused baseline differs at decode warm-up')
        if (refenv.get('DS4_CUDA_QWEN_KV_CAPACITY_MODE')=='hybrid-attention')!=(a.baseline_mode=='kv-offload'):
            p.error('reused baseline differs at KV placement')
    a.output=a.output.resolve(); a.output.mkdir(parents=True,exist_ok=True)
    a.start=a.end=a.frontier; a.mtp_draft=5 if a.mtp else None
    common={'DS4_QWEN_EXTENDED_DRAFT':'off','DS4_BENCH_DECODE_TIMES':'1',
        'DS4_BENCH_PROGRESS':'1','DS4_QWEN_PREFILL_PHASES':'1'}
    if a.decode_warmup: common['DS4_BENCH_DECODE_WARMUP']=str(a.decode_warmup)
    records=[];pairs=[]
    for sample in range(a.samples):
        rows={}
        for variant in (('baseline','candidate') if sample%2==0 else ('candidate','baseline')):
            settings=common|{'DS4_CUDA_QWEN_WEIGHT_STREAMING':str(int(variant=='candidate'))}
            if variant=='candidate':
                # These legacy variables must not offload any KV or recurrence.
                settings|={'DS4_CUDA_QWEN_WEIGHT_RESIDENT_MB':str(a.resident_mb),
                    'DS4_CUDA_QWEN_KV_CAPACITY_MODE':'hybrid-attention','DS4_CUDA_QWEN_KV_RESIDENT':'64'}
            elif a.baseline_mode=='kv-offload':
                settings|={'DS4_CUDA_QWEN_KV_CAPACITY_MODE':'hybrid-attention',
                    'DS4_CUDA_QWEN_KV_RESIDENT':'32768','DS4_CUDA_QWEN_MTP_KV_MODE':'auto',
                    'DS4_CUDA_QWEN_MTP_BATCH_CATCHUP':'1','DS4_CUDA_QWEN_MTP_DECODE_ROWS':'1'}
            name=f'sample-{sample}-{variant}'
            binary=a.binary if variant=='candidate' else a.baseline or a.binary
            if variant=='baseline' and a.reuse_baseline:
                row=reference.copy();run_dir=a.reuse_baseline.resolve()
                row['reused_completed_run']=str(run_dir)
                print(f'{name}: reusing completed matched reference {run_dir}',flush=True)
            else:
                row=run_bench(a,name,binary,settings,samples=1);run_dir=a.output/name
            runtime=(run_dir/'runtime.log').read_text(errors='replace')
            row['cycles']=summarize_cycles(runtime)
            row['weight_plan']=re.findall(r'dense weight plan[^\n]+',runtime)
            row['weight_stats']=re.findall(r'dense weight streaming forwards=[^\n]+',runtime)
            row['kv_offload_active']='dense KV capacity frontier=' in runtime
            row['host_recurrent_rows']='Qwen verifier host snapshots=' in runtime
            row['memory_reports']=re.findall(r'CUDA memory report[^\n]+',runtime)
            if variant=='candidate':
                assert row['weight_plan'] and row['weight_stats'], 'weight streaming did not execute'
                assert not row['kv_offload_active'] and not row['host_recurrent_rows'], 'KV/recurrence left VRAM'
                assert a.resident_mb==0 or 'offloaded_layers=0/' not in row['weight_plan'][0], 'forced complement not streamed'
            if a.mtp: assert row['cycles']['multi_token_calls']>0, 'NextN5 verification did not execute'
            records.append(row|{'sample':sample,'variant':variant});rows[variant]=run_dir
            write_json(a.output/'report.json',dict(status='IN_PROGRESS',records=records,pairs=pairs))
        quality=compare_logits(rows['baseline']/'logits',rows['candidate']/'logits')
        pair=dict(sample=sample,quality=quality,all_pass=all(q['bit_exact'] for q in quality))
        pairs.append(pair)
        write_json(a.output/'report.json',dict(status='COMPLETE' if sample+1==a.samples else 'IN_PROGRESS',
            records=records,pairs=pairs,all_pass=all(p['all_pass'] for p in pairs),
            scope='matched diagnostic' if a.samples<5 else 'five alternating matched pairs'))
        assert pair['all_pass'], 'complete vocabulary comparison failed; preserve original tolerances'

if __name__=='__main__':main()
