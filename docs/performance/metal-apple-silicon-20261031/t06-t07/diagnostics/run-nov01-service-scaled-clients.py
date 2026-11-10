"""Resume serial client gates only after the legacy owner completes cleanup."""
import argparse,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path('.').resolve()))
from tools.qualify_metal_command import interrupt_on_termination
from tools.qualify_metal_server import run
root=Path('build/metal-qualification/t06-t07-20261031');python='agent/.venv/bin/python'

def server(family,name,command,*,timeout=600,phase=False,vision=False,seed=None,tool_start=None):
 output=root/name
 if output.exists():raise RuntimeError(f'Existing evidence must be inspected, not overwritten: {output}')
 args=argparse.Namespace(family=family,output=output,suite_config=Path('config/native-suite-metal.example.json'),resource_metrics=None,phase_profile=True,kv_seed=seed,vision=vision,mtp=False,phase_server=phase,port=18080,timeout=timeout,timeout_multiplier=4,tool_start_timeout=tool_start,max_footprint_gib=0,reason='Explicit local4xclient AND suiteoperation timeout after missing service300s limit was identified. One real8192/64 target-only streamedLLM at a time;2GiBhostreserve and2GiBSSDweightcache. Unchanged assertions/recipe/frozen numerical thresholds; separate old failures and model limitations retained.',command=command)
 print('START',name,flush=True);report=run(args);print('FINISHED',name,report['status'],round(report['elapsed_seconds'],3),flush=True)

with interrupt_on_termination():
 deadline=time.monotonic()+14400
 while not (root/'nov01-final-clients-x4-finished.txt').exists():
  if time.monotonic()>deadline:raise TimeoutError('Legacy qualification owner did not finish cleanup')
  time.sleep(2)
 accepted=subprocess.run(['build/metal-venv/bin/python',str(root/'analyze-latest-acceptance.py')],check=False)
 if accepted.returncode:raise RuntimeError('Frozen numerical acceptance must remain PASS before client qualification')
 for family in ('flash','qwen27'):
  server(family,f'{family}-phase-cancel-service-v2',[python,'-m','pytest','-q','-s','tests/test_suite_phase_live.py'],phase=True)
 server('flash','flash-interactive-full-service-v2',[python,'-m','pytest','-q','-rs','tests/test_interactive_api.py::InteractiveAPI'],timeout=1800)
 server('flash','flash-frontend-file-create-service-v2',['env','PYTHONPATH=agent/src:agent',python,'-m','pytest','-n','0','--no-cov','-q','-s','agent/tests/test_live_e2e.py::test_live_e2e_interactive_file_creation'],timeout=1200,seed=root/'flash-frontend-file-create-v2/kv')
 server('qwen27','qwen27-frontend-single-skill-service-v2',['env','PYTHONPATH=agent/src:agent',python,'-m','pytest','-n','0','--no-cov','-q','-s','agent/tests/test_live_e2e.py::test_live_e2e_single_skill_file_creation'],timeout=1200,seed=root/'qwen27-frontend-file-create-x4/kv')
 server('qwen27','qwen27-exmekh-registered-image-service-v2',[python,'-m','pytest','-q','-s','tests/test_suite_exmekh_live.py::ExmekhNativeLive::test_registered_image_tool_retains_conversation'],timeout=1200,seed=root/'qwen27-exmekh-registered-image-v3/kv')
 server('qwen27','qwen27-browser-image-service-v2',['env','PYTHONPATH=agent/src:agent','DS4_SUITE_E2E_IMAGE_COUNT=1','DS4_SUITE_E2E_COMPACT_REPLY=1',python,'-m','pytest','-n','0','--no-cov','-q','-s','agent/tests/test_native_image_generation_e2e.py::test_real_image_generation_delivers_pixels_to_model_and_persists_in_ui'],timeout=1800,vision=True,seed=root/'qwen27-browser-image-measured-budget/kv',tool_start=1000)
 (root/'nov01-service-scaled-clients-finished.txt').write_text('All actual client owners completed cleanup; inspect every individual verdict and JUnit. Model-limit skips remain NOT_VERIFIED.\n')
