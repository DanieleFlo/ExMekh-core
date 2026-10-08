#!/usr/bin/env python3
"""Compare real extended-context greedy target decode and Qwen support models.

This does not label NextN as DSpark. Proposals must reproduce the complete
target continuation; full-vocabulary differences remain in the report.
"""
import argparse
import json
import re
import subprocess
import statistics
import time
from pathlib import Path
from perf_qwen_flash_strata import ROOT, environment, digest, write_json, summarize_csv, compare_logits


def summarize_cycles(runtime):
    """Keep untimed proposal cycles out of acceptance and cost measurements."""
    runtime = runtime.rsplit('ds4-bench: untimed decode warm-up completed=', 1)[-1]
    cycles, pending, measured = [], None, []
    for line in runtime.splitlines():
        if line.startswith('QWEN_MTP_CYCLE '):
            pending = {key: float(value) for key, value in
                       re.findall(r'(\w+)=(-?\d+(?:\.\d+)?)', line)}
        elif line.startswith('ds4-bench: decode cycle '):
            call = {key: float(value) for key, value in
                    re.findall(r'(\w+)=(-?\d+(?:\.\d+)?)', line)}
            measured.append(call)
            if pending is not None:
                cycles.append(pending)
                pending = None
    if not measured:
        raise ValueError('missing measured decode cycles')
    result = {'measured_calls': len(measured),
              'multi_token_calls': sum(call['rows'] > 1 for call in measured),
              'committed_tokens': int(sum(call['rows'] for call in measured)),
              'decode_ms': sum(call['ms'] for call in measured),
              'verifier_cycles': len(cycles)}
    result['tokens_per_second'] = 1000 * result['committed_tokens'] / result['decode_ms']
    if cycles:
        proposed = sum(cycle['drafted_tokens'] for cycle in cycles)
        accepted = sum(cycle['accepted_tokens'] for cycle in cycles)
        result.update(proposed_tokens=int(proposed), accepted_draft_tokens=int(accepted),
                      acceptance_fraction=accepted / proposed if proposed else None,
                      mean_tokens_per_verifier_cycle=statistics.mean(cycle['committed_tokens'] for cycle in cycles),
                      full_accept_cycles=sum(cycle.get('full_accept', 0) for cycle in cycles))
        result['cost_ms'] = {key: {'total': sum(cycle.get(key, 0) for cycle in cycles),
                                 'mean': statistics.mean(cycle.get(key, 0) for cycle in cycles)}
                             for key in ('draft_propose_ms', 'target_verify_ms', 'snapshot_ms',
                                         'rollback_ms', 'replay_ms', 'sidecar_catchup_ms', 'total_ms')}
        result['cycles'] = cycles
    return result


def summarize_pairs(records, widths, frontier):
    base = {row['sample']: row for row in records if row['proposals'] == 0}
    summary = []
    for width in widths:
        rows = [row for row in records if row['proposals'] == width]
        entry = {'proposals': width, 'samples': len(rows)}
        for metric in ('prefill', 'decode', 'decode_total'):
            values = [row['frontiers'][str(frontier)][metric]['median'] for row in rows]
            reference = [base[row['sample']]['frontiers'][str(frontier)][metric]['median'] for row in rows]
            changes = [100 * (a/b - 1) for a, b in zip(values, reference)]
            entry[metric] = {'samples': values, 'median': statistics.median(values),
                             'paired_change_percent': changes,
                             'median_change_percent': statistics.median(changes),
                             'min_change_percent': min(changes),
                             'cv': statistics.stdev(values)/statistics.mean(values) if len(values) > 1 else None}
        summary.append(entry)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, default=ROOT / 'ds4-bench')
    p.add_argument('--model', type=Path, default=ROOT / 'gguf/Qwen3.8-27B-UD-Q4_K_S.gguf')
    p.add_argument('--mtp', type=Path, default=ROOT / 'gguf/mtp-Qwen3.8-27B-Q4_0.gguf')
    p.add_argument('--drafter', choices=['nextn', 'dspark'], default='nextn')
    p.add_argument('--confidence', type=float, help='Explicit DSpark confidence threshold, including 0 to disable pruning')
    p.add_argument('--chunk', type=int, default=1024)
    p.add_argument('--prompt', type=Path, default=ROOT / 'speed-bench/promessi_sposi.txt')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frontier', type=int, default=65536)
    p.add_argument('--resident', type=int, default=32768)
    p.add_argument('--mtp-kv-mode', choices=['shared', 'auto', 'resident', 'staged'], default='shared')
    p.add_argument('--batch-catchup', choices=['off', 'on'], default='off')
    p.add_argument('--tokens', type=int, default=64)
    p.add_argument('--width', action='append', type=int)
    p.add_argument('--samples', type=int, default=1)
    p.add_argument('--warmup', type=int, default=0)
    p.add_argument('--decode-warmup', type=int, default=0, help='Untimed decode tokens restored with compact prefix state')
    p.add_argument('--timeout', type=int, default=3600)
    p.add_argument('--cycle-trace', choices=['auto', 'on', 'off'], default='auto',
                   help='Auto traces diagnostics; five-pair measurements disable cycle logging')
    p.add_argument('--summarize-only', action='store_true', help='Add cycle costs to an existing report without rerunning inference')
    p.add_argument('--resume', action='store_true', help='Reuse completed runs with exactly matching binary, inputs and controls')
    p.add_argument('--prefill-phases', action='store_true', help='Diagnostic GPU fences separate target prefill and draft preparation')
    a = p.parse_args()
    if a.summarize_only:
        path = a.output.resolve() / 'report.json'
        report = json.loads(path.read_text())
        for row in report['records']:
            if row.get('status') == 'COMPLETE':
                runtime = (Path(row['output']) / 'runtime.log').read_text(errors='replace')
                row['cycle_summary'] = summarize_cycles(runtime)
                warmup = int(row['command'][row['command'].index('--repetitions')+1]) - 1
                row['frontiers'] = summarize_csv(Path(row['output']) / 'speed.csv', warmup)
        write_json(path, report)
        return
    widths = a.width or [2, 7, 14, 15]
    if a.frontier <= a.resident or any(w < 2 or w > 15 for w in widths):
        p.error('require real remote KV and NextN width 2..15')
    if a.samples < 1 or a.tokens < 1 or a.decode_warmup < 0 or a.decode_warmup > a.tokens:
        p.error('require positive samples/tokens and decode warm-up within the generated token count')
    if a.prefill_phases and a.samples >= 5:
        p.error('prefill phase profiling is diagnostic; disable it for paired promotion measurements')
    a.output = a.output.resolve(); a.output.mkdir(parents=True, exist_ok=True)
    identity = digest(a.binary)
    extra = {'DS4_CUDA_QWEN_KV_CAPACITY_MODE': 'hybrid-attention',
             'DS4_CUDA_QWEN_KV_RESIDENT': str(a.resident),
             'DS4_CUDA_QWEN_MTP_KV_MODE': a.mtp_kv_mode,
             'DS4_CUDA_QWEN_MTP_BATCH_CATCHUP': '1' if a.batch_catchup == 'on' else '0',
             'DS4_BENCH_PROGRESS': '1', 'DS4_BENCH_DECODE_TIMES': '1',
             'DS4_QWEN_EXTENDED_DRAFT': 'off'}
    trace = a.cycle_trace == 'on' or (a.cycle_trace == 'auto' and a.samples < 5)
    if trace: extra.update(DS4_BENCH_SPEC_TRACE='1', DS4_MTP_CYCLE_TRACE='1')
    if a.decode_warmup: extra['DS4_BENCH_DECODE_WARMUP']=str(a.decode_warmup)
    if a.prefill_phases: extra['DS4_QWEN_PREFILL_PHASES']='1'
    env = environment(extra)
    previous = {}
    if a.resume and (a.output / 'report.json').is_file():
        previous = {(row['sample'], row['proposals']): row for row in
                    json.loads((a.output / 'report.json').read_text())['records']}
    records, pairs = [], []
    for sample in range(a.samples):
        order = [0, *widths] if sample % 2 == 0 else [*reversed(widths), 0]
        for width in order:
            out = a.output / f'sample-{sample}-{a.drafter}-{width}'
            out.mkdir(parents=True, exist_ok=True)
            logits = out / 'logits'; logits.mkdir(exist_ok=True)
            command = [str(a.binary.resolve()), '--cuda', '-m', str(a.model.resolve()),
                '--prompt-file', str(a.prompt.resolve()), '--ctx-start', str(a.frontier),
                '--ctx-max', str(a.frontier), '--ctx-alloc', str(a.frontier + a.tokens + 32),
                '--step-incr', '1', '--gen-tokens', str(a.tokens), '--prefill-chunk', str(a.chunk),
                '--repetitions', str(a.warmup + 1), '--csv', str(out / 'speed.csv'),
                '--dump-frontier-logits-dir', str(logits)]
            if width: command += ['--mtp', '--mtp-model', str(a.mtp.resolve()), '--mtp-draft', str(width)]
            if width and a.confidence is not None:
                if a.drafter != 'dspark': p.error('confidence applies to DSpark')
                command += ['--dspark-confidence',str(a.confidence)]
            saved = previous.get((sample, width))
            if saved and saved.get('status') == 'COMPLETE':
                saved_env = dict(saved['environment'])
                saved_env.setdefault('DS4_CUDA_QWEN_MTP_BATCH_CATCHUP', '0')
                if saved['binary_sha256'] != identity or saved['command'] != command or saved_env != extra:
                    raise ValueError('resume inputs changed; use a separate output directory')
                saved['cycle_summary'] = summarize_cycles((out / 'runtime.log').read_text(errors='replace'))
                saved['frontiers'] = summarize_csv(out / 'speed.csv', a.warmup)
                records.append(saved)
                print(f'reuse context {a.frontier}, {a.drafter}={width}, sample={sample}', flush=True)
                continue
            row = {'sample': sample, 'drafter': a.drafter if width else 'target-only', 'proposals': width, 'status': 'RUNNING',
                'command': command, 'environment': extra, 'binary_sha256': identity}
            records.append(row); write_json(a.output / 'report.json', {'records': records, 'pairs': pairs})
            print(f'actual context {a.frontier}, {a.drafter}={width}, sample={sample}', flush=True)
            if digest(a.binary) != identity: raise RuntimeError('benchmark binary changed before inference')
            started = time.monotonic()
            with (out / 'runtime.log').open('wb') as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=log, timeout=a.timeout)
            if digest(a.binary) != identity: raise RuntimeError('benchmark binary changed during inference')
            row.update(exit_code=result.returncode, wall_seconds=time.monotonic() - started,
                       status='COMPLETE' if result.returncode == 0 else 'FAILED')
            if result.returncode:
                write_json(a.output / 'report.json', {'records': records, 'pairs': pairs})
                raise RuntimeError('decode failed; retained log')
            row['frontiers'] = summarize_csv(out / 'speed.csv', a.warmup)
            row['output'] = str(out)
            runtime = (out / 'runtime.log').read_text(errors='replace')
            phase_rows = [dict((key, float(value)) for key, value in
                          re.findall(r'(\w+)=(-?\d+(?:\.\d+)?)', line))
                          for line in runtime.splitlines() if line.startswith('QWEN_PREFILL_PHASE ')]
            if phase_rows:
                row['prefill_phases'] = {key: sum(phase[key] for phase in phase_rows)
                    for key in ('rows', 'target_ms', 'draft_prepare_ms')}
                row['prefill_phases']['chunks'] = phase_rows
            row['cycle_summary'] = summarize_cycles(runtime)
            row['actual_cycles'] = runtime.count('QWEN_MTP_CYCLE cycle=') if trace else None
            row['actual_verifier_cycles'] = sum(int(n)>0 for n in re.findall(r'QWEN_MTP_CYCLE[^\n]*drafted_tokens=(\d+)',runtime)) if trace else None
            row['confidence_bypasses'] = runtime.count('confidence_bypass=1')
            if width:
                if trace: assert row['actual_cycles'] > 0, 'speculation did not execute real verifier cycles'
                else:
                    assert 'NextN MTP support model loaded:' in runtime or 'Qwen DSpark v2 loaded:' in runtime, 'support model did not load'
                    assert row['cycle_summary']['multi_token_calls'] > 0, 'no measured multi-token speculative calls'
                assert 'falling back' not in runtime and 'future cycles will use target-only' not in runtime, 'support runtime fell back'
                if a.drafter == 'dspark': assert 'Qwen DSpark v2 loaded:' in runtime, 'wrong support model'
        base = a.output / f'sample-{sample}-{a.drafter}-0/logits'
        base_end = json.loads((base / f'frontier_{a.frontier:06d}.decode.json').read_text())
        assert base_end['generated_count'] == a.tokens and base_end['final_position'] == a.frontier + a.tokens
        for width in widths:
            candidate = a.output / f'sample-{sample}-{a.drafter}-{width}/logits'
            end = json.loads((candidate / f'frontier_{a.frontier:06d}.decode.json').read_text())
            same_tokens = end['generated_tokens'] == base_end['generated_tokens']
            quality = compare_logits(base, candidate) if same_tokens else []
            pairs.append({'sample': sample, 'drafter': a.drafter, 'proposals': width,
                'same_generated_tokens': same_tokens,
                'actual_final_position': end['final_position'], 'quality': quality})
        passed = all(pair['same_generated_tokens'] and pair['quality'] and
                     all(q['bit_exact'] for q in pair['quality']) for pair in pairs)
        write_json(a.output / 'report.json', {'records': records, 'pairs': pairs,
            'summary': summarize_pairs(records, widths, a.frontier),
            'status': 'COMPLETE', 'scope': 'diagnostic' if a.samples < 5 else 'paired measurements',
            'all_pass': passed})
        if not passed: raise RuntimeError('target continuation or complete logits differ; retained quality report')


if __name__ == '__main__': main()
