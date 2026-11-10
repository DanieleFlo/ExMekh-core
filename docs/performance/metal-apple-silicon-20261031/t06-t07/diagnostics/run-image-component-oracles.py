"""Independent CPU encoder/DiT and MPS VAE, with explicit per-stage evidence."""
import hashlib,json,os,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path('.').resolve()))
from tools.qualify_metal_command import interrupt_on_termination,run
from tools.qualify_qwen_image_pipeline import main as qualify
root=Path('build/metal-qualification/t06-t07-20261031')
os.environ['DS4_METAL_TIMEOUT_MULTIPLIER']='4'
outcomes=[]
def arguments(output,case):
 return ['tools/qualify_qwen_image_pipeline.py','--prepared','build/metal-oracles','--bundle','build/metal-models/image-bf16.ds4b/bundle.ds4b','--tokenizer','build/metal-models/image-tokenizer.ds4q','--output',str(output),'--bounded','--case-ids',case,'--oracle-device','cpu','--oracle-vae-device','mps','--oracle-offload']
with interrupt_on_termination():
 for case in ('text','reference','rgba'):
  output=root/f'product-image-{case}-cpu-vae-mps'
  if not (output/'functional.json').exists():
   print('START latest native image',case,flush=True)
   sys.argv=arguments(output,case)+['--native-only']
   qualify()
  devices=dict(encoder='cpu',dit='cpu',encode='mps',decode='mps')
  reused={}
  if case=='text' and not (output/'oracle/text/latent-7.f32').exists():
   previous=root/'product-image-text-cpu'
   old=json.loads((previous/'cases.json').read_text())[0]
   new=json.loads((output/'cases.json').read_text())[0]
   if old['fixture_sha256']!=new['fixture_sha256']:raise ValueError('reused CPU oracle input changed')
   for stage in ('encoder','encode','dit'):
    report=json.loads((previous/f'run-{stage}/report.json').read_text())
    if report['status']!='PASS':raise ValueError('incomplete CPU oracle stage')
    reused[stage]=str(previous/f'run-{stage}/report.json')
   shutil.copytree(previous/'oracle',output/'oracle',dirs_exist_ok=True)
  for stage in ('encoder','encode','dit','decode'):
   if stage in reused:continue
   folder=output/f'run-oracle-{stage}'
   if (folder/'report.json').exists():
    report=json.loads((folder/'report.json').read_text())
    if report['status']!='PASS':raise ValueError('existing failed oracle requires a separate retry directory')
    continue
   print('START oracle',case,stage,devices[stage],flush=True)
   command=[sys.executable,'tools/run_qwen_image_pipeline_oracle.py','--prepared','build/metal-oracles','--cases',str(output/'cases.json'),'--output',str(output/'oracle'),'--stage',stage,'--device',devices[stage],'--offload',*(['--trace-encoder'] if stage=='encoder' else [])]
   report=run(command,folder,300,0,'Independent pinned HF/Diffusers component reference: CPU encoder/DiT, MPS VAE, unchanged BF16 recipe/noise/8steps/256px and frozen acceptance thresholds. Prior scalar CPU VAE attempt retained NOT_VERIFIED after measured slow_conv2d. Serial models,2GiBhostreserve,user4x deadlines.',timeout_multiplier=4)
   print('FINISHED oracle',case,stage,report['status'],flush=True)
   if report['status']!='PASS':raise RuntimeError('reference stage did not complete')
  profile=dict(devices=devices,reused_stages=reused,scope='Explicit independent component-device profile; CPU-only VAE remains NOT_VERIFIED. No precision/recipe/tolerance changes.')
  if reused:profile['devices']['encode']='cpu'
  (output/'oracle-profile.json').write_text(json.dumps(profile,indent=2)+'\n')
  print('ANALYZE',case,flush=True)
  sys.argv=arguments(output,case)+['--skip-native','--skip-oracle']
  try:qualify()
  except SystemExit as error:
   if error.code not in (0,1):raise
  result=json.loads((output/'verification.json').read_text())
  result['oracle_devices']=profile['devices']
  result['oracle_profile']=profile
  (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
  outcomes.append(dict(case=case,status=result['verdict'],path=str(output)))
  print('FINISHED',case,result['verdict'],flush=True)
 (root/'product-images-cpu-vae-mps-finished.json').write_text(json.dumps(outcomes,indent=2)+'\n')
