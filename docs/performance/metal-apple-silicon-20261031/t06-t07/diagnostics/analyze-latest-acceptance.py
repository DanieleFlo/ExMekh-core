"""Analyze frozen image/media gates before allowing the long client queue."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path('.').resolve()))
from tools.qualify_metal_suite import compare_boundaries, compare_vectors

root = Path('build/metal-qualification/t06-t07-20261031')
folder = root / 'product-bf16-media-v4'
hidden = compare_boundaries(folder / 'native', root / 'gemma-media/oracle')
vectors = compare_vectors(folder / 'native/vectors.f32', root / 'gemma-media/oracle/vectors.npy', 128)
images = json.loads((root / 'product-images-cpu-oracles-finished.json').read_text())
passed = hidden['status'] == vectors['status'] == 'PASS' and all(case['status'] == 'PASS' for case in images['cases'])
report = dict(status='PASS' if passed else 'FAIL', internal_boundaries=hidden, output_vectors=vectors, images=images, scope='Unchanged frozen strict gates on current saved native media and explicit independent CPU encoder/DiT plus MPS VAE image profile. Prior failures retained separately.')
(folder / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
print('media', hidden['status'], hidden['exact'], '/', hidden['count'], 'vectors', vectors['status'], 'images', [(case['case'], case['status']) for case in images['cases']], flush=True)
raise SystemExit(0 if passed else 1)
