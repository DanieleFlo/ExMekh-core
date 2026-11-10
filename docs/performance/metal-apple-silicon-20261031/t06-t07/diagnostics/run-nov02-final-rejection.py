"""Final literal-rejection consumer after the sole large-model queue closes."""
import argparse,sys,time
from pathlib import Path
sys.path.insert(0,str(Path('.').resolve()))
from tools.qualify_metal_command import interrupt_on_termination
from tools.qualify_metal_server import run
root=Path('build/metal-qualification/t06-t07-20261031')
with interrupt_on_termination():
 deadline=time.monotonic()+43200
 while not (root/'nov01-service-scaled-clients-finished.txt').exists():
  if time.monotonic()>deadline:raise TimeoutError('Prior model owner did not finish cleanup')
  time.sleep(2)
 args=argparse.Namespace(family='qwen27',output=root/'qwen27-interactive-reject-template-service-v2',suite_config=Path('config/native-suite-metal.example.json'),resource_metrics=None,phase_profile=True,kv_seed=None,vision=False,mtp=False,phase_server=False,port=18080,timeout=600,timeout_multiplier=4,tool_start_timeout=None,max_footprint_gib=0,reason='Original rejection assertions, explicit literal template; no skip promotion. One8k/64target-onlyQwen,2GiBcache/reserve,allclient/service/interactive4x.',command=['agent/.venv/bin/python','-m','pytest','-q','-s','tests/test_interactive_api.py::InteractiveAPI::test_second_resource_rejection_with_literal_template'])
 report=run(args);print('FINISHED literal rejection',report['status'],flush=True)
 (root/'nov01-frontend-edit-delete-finished.txt').write_text('All actual Custom file/frontend/SDK/browser owners closed in predecessor; final literal-rejection owner now closed. Inspect each verdict. Generic originals remain separate.\n')
