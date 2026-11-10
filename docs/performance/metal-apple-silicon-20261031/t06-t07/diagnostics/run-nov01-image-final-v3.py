"""Latest full native RGB/RGBA pipelines, reusing byte-identical independent oracles."""
import hashlib,json,os,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path('.').resolve()))
from tools.qualify_metal_command import interrupt_on_termination
from tools.qualify_qwen_image_pipeline import main as qualify
root=Path('build/metal-qualification/t06-t07-20261031');os.environ['DS4_METAL_TIMEOUT_MULTIPLIER']='4'
with interrupt_on_termination():
 cases=[]
 for case in ('reference','rgba'):
  trace=root/f'product-encoder-{case}-final-v3/native/conditioning-0.bf16'
  if not trace.exists():raise RuntimeError('Independent complete encoder check must finish first')
  previous=root/f'product-image-{case}-cpu-vae-mps';output=root/f'product-image-{case}-cpu-vae-mps-v3'
  arguments=['qualify','--prepared','build/metal-oracles','--bundle','build/metal-models/image-bf16.ds4b/bundle.ds4b','--tokenizer','build/metal-models/image-tokenizer.ds4q','--output',str(output),'--bounded','--case-ids',case,'--oracle-device','cpu','--oracle-vae-device','mps','--oracle-offload']
  print('START complete latest pipeline',case,flush=True);sys.argv=arguments+['--native-only'];qualify()
  old=json.loads((previous/'cases.json').read_text())[0];new=json.loads((output/'cases.json').read_text())[0]
  if old['fixture_sha256']!=new['fixture_sha256']:raise ValueError('Independent reference reuse requires identical serialized inputs and noise')
  shutil.copytree(previous/'oracle',output/'oracle',dirs_exist_ok=True)
  profile=json.loads((previous/'oracle-profile.json').read_text());profile.update(reused_from=str(previous),fixture_sha256=new['fixture_sha256'],native_binary_sha256=hashlib.sha256(Path('tests/probe_qwen_image_pipeline').read_bytes()).hexdigest());(output/'oracle-profile.json').write_text(json.dumps(profile,indent=2)+'\n')
  sys.argv=arguments+['--skip-native','--skip-oracle']
  try:qualify()
  except SystemExit as error:
   if error.code not in (0,1):raise
  report=json.loads((output/'verification.json').read_text());cases.append(dict(case=case,status=report['verdict'],path=str(output)))
  print('FINISHED complete pipeline',case,report['verdict'],flush=True)
 text=root/'product-image-text-cpu-vae-mps';report=json.loads((text/'verification.json').read_text());cases.insert(0,dict(case='text',status=report['verdict'],path=str(text),reuse_scope='Completed full text generation. Subsequent production changes concern only visual encoder erf/position/trig and audio channelRMS; text encoder, DiT and VAE paths unchanged.'))
 marker=dict(scope='All serial native/reference owners completed cleanup; explicit independent CPU encoder/DiT and MPS VAE. Scalar CPU-only VAE remains NOT_VERIFIED. RGB/RGBA reference inputs and noise SHA verified identical.',cases=cases)
 (root/'product-images-cpu-oracles-finished.json').write_text(json.dumps(marker,indent=2)+'\n')
 print('CLOSED image acceptance',[(case['case'],case['status']) for case in cases],flush=True)
