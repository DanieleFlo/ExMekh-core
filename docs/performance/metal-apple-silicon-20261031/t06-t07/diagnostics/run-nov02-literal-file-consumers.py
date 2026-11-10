"""One explicit literal-framing case only where the original file workflow is unverified."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path('.').resolve()))
from tools.qualify_metal_command import interrupt_on_termination
from tools.qualify_metal_server import run
root=Path('build/metal-qualification/t06-t07-20261031')
with interrupt_on_termination():
 deadline=time.monotonic()+43200
 while not (root/'nov01-frontend-edit-delete-finished.txt').exists():
  if time.monotonic()>deadline:raise TimeoutError('Prior real client owners did not close')
  time.sleep(2)
 records=[]
 for family in ('flash','qwen27'):
  original=root/f'{family}-frontend-custom-files-sdk-budget-v1'
  prior=json.loads((original/'report.json').read_text())
  if prior['status']=='PASS' and prior.get('all_selected_tests_passed') is True:
   records.append({'family':family,'original_report':str(original/'report.json'),'original_status':'PASS','additional_execution_started':False,'reason':'Complete original workflow already verified; no extra repetition.'})
   continue
  output=root/f'{family}-frontend-custom-literal-files-sdk-budget-v1'
  if output.exists():raise RuntimeError(f'Preserve existing evidence: {output}')
  args=argparse.Namespace(family=family,output=output,suite_config=Path('config/native-suite-metal.example.json'),resource_metrics=None,phase_profile=True,kv_seed=original/'kv',vision=False,mtp=False,phase_server=False,port=18080,timeout=1800,timeout_multiplier=4,tool_start_timeout=None,max_footprint_gib=0,reason='One DISTINCT explicit literal-framing input after original workflow did not verify exact no-newline contents. Original contract retains all leading/trailing bytes by design; no trimming, no assertion or expected-result changes. Same public Custom AgentDesign and original real file tools; create->inspect/read->edit->delete, one8192/64targetonly model,2GiBcache/reserve,alltimeouts4x.',command=['env','PYTHONPATH=agent/src:agent','agent/.venv/bin/python','-m','pytest','-n','0','--no-cov','-q','-s','agent/tests/test_live_e2e.py::test_live_e2e_custom_literal_file_workflow'])
  print('START',family,'literal file workflow',flush=True)
  result=run(args);print('FINISHED',family,result['status'],flush=True)
  records.append({'family':family,'original_report':str(original/'report.json'),'original_status':prior['status'],'additional_execution_started':True,'report':str(output/'report.json'),'status':result['status']})
 (root/'nov02-literal-file-consumers-finished.json').write_text(json.dumps({'scope':'All planned literal-framing owners closed; this marker does not promote individual verdicts.','records':records},indent=2)+'\n')
