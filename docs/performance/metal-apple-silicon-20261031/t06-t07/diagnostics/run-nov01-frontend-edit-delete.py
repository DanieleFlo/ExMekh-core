"""Remaining literal-rejection and explicit Custom file workflows, strictly serial."""
import argparse,sys,time
from pathlib import Path
sys.path.insert(0,str(Path('.').resolve()))
from tools.qualify_metal_command import interrupt_on_termination
from tools.qualify_metal_server import run
root=Path('build/metal-qualification/t06-t07-20261031')

def server(family,name,test,seed=None,*,frontend=True,timeout=1800):
 print('START',name,flush=True)
 command=['agent/.venv/bin/python','-m','pytest','-q','-s',test]
 if frontend:command=['env','PYTHONPATH=agent/src:agent',*command[:4],'-n','0','--no-cov',*command[4:]]
 args=argparse.Namespace(family=family,output=root/name,suite_config=Path('config/native-suite-metal.example.json'),resource_metrics=None,phase_profile=True,kv_seed=seed,vision=False,mtp=False,phase_server=False,port=18080,timeout=timeout,timeout_multiplier=4,tool_start_timeout=None,max_footprint_gib=0,reason='Actual unchanged rejection assertions with explicit literal-template input, then explicit public Custom profile: real create, inspect/read, edit and delete with the same original tool/effect/history checks; Generic profile model limitations remain separate. Prior vague-prompt model limitation remains NV. One8192/64target-only model,2GiBreserve,user4xservice/client/interactive budgets.',command=command)
 result=run(args);print('FINISHED',name,result['status'],round(result['elapsed_seconds'],3),flush=True)

with interrupt_on_termination():
 deadline=time.monotonic()+21600
 while not (root/'nov01-service-scaled-clients-finished.txt').exists():
  if time.monotonic()>deadline:raise TimeoutError('Prior serial model owner did not finish cleanup')
  time.sleep(2)
 server('qwen27','qwen27-interactive-reject-template-service-v2','tests/test_interactive_api.py::InteractiveAPI::test_second_resource_rejection_with_literal_template',frontend=False,timeout=600)
 for family,seed in [('flash',root/'flash-frontend-single-skill-sdk-budget-v3/kv'),('qwen27',root/'qwen27-frontend-single-skill-sdk-budget-v3/kv')]:
  server(family,f'{family}-frontend-custom-files-sdk-budget-v1','agent/tests/test_live_e2e.py::test_live_e2e_custom_file_workflow',seed)
 (root/'nov01-frontend-edit-delete-finished.txt').write_text('All rejection/Custom frontend create/edit/delete owners completed cleanup. Inspect per-case JUnit, not only process exit.\n')
