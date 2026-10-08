#!/usr/bin/env python3
"""Convert the pinned official Qwen3.8-27B DSpark v2 support checkpoint.

Target weights, tokenizer, and precision are untouched. Linear draft weights
may use DS4's existing quantizer; small norms/confidence tensors remain F32.
The resulting file contains a support model, not an independent target model.
"""
import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
import numpy as np
from glm53_quantize import (Quantizer, align, pack_string, kv_string, kv_u32,
                           kv_u32_array, kv_f32, qtype_nbytes)

REVISION = 'b9a5dbdf03bc999c6c73c426b19c2d9041cea393'
SOURCE_SHA256 = '2aff025f45823b40ebe726b9dfa40302f3512bd9a11c3a7347de32a567acd9a7'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(16 << 20), b''): h.update(block)
    return h.hexdigest()


def schema():
    result = {'fc.weight': [5120, 25600], 'hidden_norm.weight': [5120],
        'norm.weight': [5120], 'markov_head.markov_w1.weight': [248320, 256],
        'markov_head.markov_w2.weight': [248320, 256],
        'confidence_head.proj.weight': [1, 5376], 'confidence_head.proj.bias': [1]}
    for i in range(5):
        prefix = f'layers.{i}.'
        result.update({prefix + name: shape for name, shape in {
            'input_layernorm.weight': [5120], 'post_attention_layernorm.weight': [5120],
            'self_attn.q_norm.weight': [128], 'self_attn.k_norm.weight': [128],
            'self_attn.q_proj.weight': [4096, 5120], 'self_attn.k_proj.weight': [1024, 5120],
            'self_attn.v_proj.weight': [1024, 5120], 'self_attn.o_proj.weight': [5120, 4096],
            'mlp.gate_proj.weight': [17408, 5120], 'mlp.up_proj.weight': [17408, 5120],
            'mlp.down_proj.weight': [5120, 17408]}.items()})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--quant', choices=['f16', 'q8_0', 'q4_k'], default='q8_0')
    p.add_argument('--library', type=Path, default=Path(__file__).with_name('libds4quants.so'))
    p.add_argument('--rows-per-chunk', type=int, default=128)
    p.add_argument('--inspect', action='store_true', help='Validate and print layout without conversion')
    a = p.parse_args()
    if a.rows_per_chunk < 1: p.error('rows-per-chunk must be positive')
    cfg = json.loads((a.source / 'config.json').read_text())
    for key, expected in {'architectures': ['DSparkDraftModel'], 'hidden_size': 5120,
        'num_hidden_layers': 5, 'num_attention_heads': 32, 'num_key_value_heads': 8,
        'head_dim': 128, 'intermediate_size': 17408, 'vocab_size': 248320,
        'target_layer_ids': [5, 19, 33, 47, 61], 'markov_rank': 256,
        'markov_head_type': 'vanilla', 'mask_token_id': 248070,
        'block_size': 7, 'training_block_size': 16}.items():
        if cfg.get(key) != expected: raise ValueError(f'unsupported {key}: {cfg.get(key)}')
    rope = cfg['rope_parameters']
    if rope != {'beta_fast': 32.0, 'beta_slow': 1.0, 'factor': 32.0,
                'original_max_position_embeddings': 8192, 'rope_theta': 10000000,
                'rope_type': 'yarn'}:
        raise ValueError('unsupported rotary configuration')
    source = a.source / 'model.safetensors'
    with source.open('rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        if header_size > 1 << 20: raise ValueError('unreasonable tensor header')
        header = json.loads(f.read(header_size))
    expected = schema()
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    if set(tensors) != set(expected): raise ValueError('unexpected checkpoint tensors')
    linear_type = {'f16': 1, 'q8_0': 8, 'q4_k': 12}[a.quant]
    plan, offset = [], 0
    ranges = []
    for name, info in sorted(tensors.items()):
        shape = info['shape']; start, end = info['data_offsets']
        if info['dtype'] != 'BF16' or shape != expected[name] or end-start != int(np.prod(shape))*2:
            raise ValueError(f'invalid tensor: {name}')
        if start < 0 or end + header_size + 8 > source.stat().st_size:
            raise ValueError('truncated source')
        ranges.append((start, end))
        qt = 0 if len(shape) == 1 or name.startswith('confidence_head.') else (
            1 if name == 'markov_head.markov_w1.weight' else linear_type)
        dimensions = tuple(reversed(shape))
        size = qtype_nbytes(qt, dimensions)
        offset = align(offset)
        plan.append((name, dimensions, qt, offset, size, start, shape))
        offset += size
    ranges.sort()
    if any(ranges[i][1] > ranges[i+1][0] for i in range(len(ranges)-1)):
        raise ValueError('overlapping tensor ranges')
    metadata = [kv_string('general.architecture', 'qwen3-dspark'),
        kv_string('general.name', 'Qwen3.8-27B DSpark v2 support'),
        kv_string('general.source.repo', 'RadixArk/Qwen3.8-27B-DSpark'),
        kv_string('general.source.revision', REVISION),
        kv_string('general.source.sha256', SOURCE_SHA256), kv_u32('general.alignment', 32),
        kv_u32('dspark.block_size', 7), kv_u32('dspark.training_block_size', 16),
        kv_u32('dspark.markov_rank', 256), kv_u32('dspark.noise_token_id', 248070),
        kv_u32_array('dspark.target_layer_ids', [5, 19, 33, 47, 61]),
        kv_u32('qwen3-dspark.embedding_length', 5120), kv_u32('qwen3-dspark.block_count', 5),
        kv_u32('qwen3-dspark.feed_forward_length', 17408),
        kv_u32('qwen3-dspark.attention.head_count', 32),
        kv_u32('qwen3-dspark.attention.head_count_kv', 8),
        kv_u32('qwen3-dspark.attention.key_length', 128),
        kv_u32('qwen3-dspark.context_length', 262144),
        kv_f32('qwen3-dspark.attention.layer_norm_rms_epsilon', 1e-6),
        kv_f32('qwen3-dspark.rope.freq_base', 1e7),
        kv_f32('qwen3-dspark.rope.scaling.factor', 32),
        kv_u32('qwen3-dspark.rope.scaling.original_context_length', 8192),
        kv_f32('qwen3-dspark.rope.scaling.beta_fast', 32),
        kv_f32('qwen3-dspark.rope.scaling.beta_slow', 1)]
    tensor_records = [pack_string(name) + struct.pack('<I', len(dims)) +
        struct.pack(f'<{len(dims)}Q', *dims) + struct.pack('<IQ', qt, pos)
        for name, dims, qt, pos, *_ in plan]
    prefix = b'GGUF' + struct.pack('<IQQ', 3, len(plan), len(metadata)) + b''.join(metadata + tensor_records)
    data_start = align(len(prefix))
    print(json.dumps({'source_revision': REVISION, 'quant': a.quant,
                      'tensors': len(plan), 'bytes': data_start + offset}), flush=True)
    if a.inspect: return
    actual = digest(source)
    if actual != SOURCE_SHA256: raise ValueError(f'checkpoint hash mismatch: {actual}')
    if a.out.exists(): raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    partial = a.out.with_suffix(a.out.suffix + '.partial')
    if partial.exists(): raise FileExistsError(partial)
    quantizer = Quantizer(str(a.library.resolve()))
    with partial.open('xb') as f:
        f.write(prefix); f.write(bytes(data_start-len(prefix)))
        for name, dims, qt, pos, size, source_pos, shape in plan:
            f.seek(data_start + pos)
            raw = np.memmap(source, mode='r', dtype='<u2', offset=8+header_size+source_pos, shape=tuple(shape))
            rows = shape[0] if len(shape) == 2 else 1
            for first in range(0, rows, a.rows_per_chunk):
                block = raw[first:first+a.rows_per_chunk] if len(shape) == 2 else raw
                floats = (np.asarray(block, dtype=np.uint32) << 16).view(np.float32)
                if not np.isfinite(floats).all(): raise ValueError(f'nonfinite weight {name}')
                f.write(quantizer.encode(floats, qt))
            del raw
            if f.tell() != data_start + pos + size: raise ValueError(f'encoded size mismatch: {name}')
            print(name, flush=True)
        f.flush(); os.fsync(f.fileno())
    partial.replace(a.out)
    manifest = {'source_revision': REVISION, 'source_sha256': actual,
        'output_sha256': digest(a.out), 'quant': a.quant, 'bytes': a.out.stat().st_size,
        'tensor_count': len(plan), 'target_weights': 'unchanged; not included'}
    a.out.with_suffix(a.out.suffix + '.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__': main()
