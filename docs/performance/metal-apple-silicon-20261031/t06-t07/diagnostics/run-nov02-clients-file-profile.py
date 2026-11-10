"""Finish real clients with explicit public file profiles and corrected coding HDS."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path('.').resolve()))
from tools.qualify_metal_command import interrupt_on_termination
from tools.qualify_metal_server import run
root=Path('build/metal-qualification/t06-t07-20261031');python='agent/.venv/bin/python'

def server(family,name,command,*,timeout=1800,vision=False,seed=None,tool_start=None):
 output=root/name
 if output.exists():raise RuntimeError(f'Existing evidence must not be overwritten: {output}')
 args=argparse.Namespace(family=family,output=output,suite_config=Path('config/native-suite-metal.example.json'),resource_metrics=None,phase_profile=True,kv_seed=seed,vision=vision,mtp=False,phase_server=False,port=18080,timeout=timeout,timeout_multiplier=4,tool_start_timeout=tool_start,max_footprint_gib=0,reason='Explicit Custom profile through public AgentDesign: real file create, inspect/read, edit/delete with original tool/effect/history checks shared with Generic. Generic model choice/old coding-HDS missing-file-skill failures retained separately. SDK97e0119e safe compaction and corrected coding resource; one8192/64target-only streamed model with2GiBcache/reserve, client/service/interactive4x.',command=command)
 print('START',name,flush=True);report=run(args);print('FINISHED',name,report['status'],round(report['elapsed_seconds'],3),flush=True)

def frontend(test):return ['env','PYTHONPATH=agent/src:agent',python,'-m','pytest','-n','0','--no-cov','-q','-s',test]

with interrupt_on_termination():
 for family,seed in [('flash',root/'flash-frontend-single-skill-sdk-budget-v3/kv'),('qwen27',root/'qwen27-frontend-file-create-x4/kv')]:
  server(family,f'{family}-frontend-custom-files-sdk-budget-v1',frontend('agent/tests/test_live_e2e.py::test_live_e2e_custom_file_workflow'),seed=seed)
 server('qwen27','qwen27-frontend-single-skill-sdk-budget-v4',frontend('agent/tests/test_live_e2e.py::test_live_e2e_single_skill_file_creation'),timeout=1200,seed=root/'qwen27-frontend-file-create-x4/kv')
 server('qwen27','qwen27-exmekh-registered-image-service-v2',[python,'-m','pytest','-q','-s','tests/test_suite_exmekh_live.py::ExmekhNativeLive::test_registered_image_tool_retains_conversation'],timeout=1200,seed=root/'qwen27-exmekh-registered-image-v3/kv')
 server('qwen27','qwen27-browser-image-service-v2',['env','PYTHONPATH=agent/src:agent','DS4_SUITE_E2E_IMAGE_COUNT=1','DS4_SUITE_E2E_COMPACT_REPLY=1',python,'-m','pytest','-n','0','--no-cov','-q','-s','agent/tests/test_native_image_generation_e2e.py::test_real_image_generation_delivers_pixels_to_model_and_persists_in_ui'],vision=True,seed=root/'qwen27-browser-image-measured-budget/kv',tool_start=1000)
 (root/'nov01-service-scaled-clients-finished.txt').write_text('All actual Custom files/Generic Qwen/SDK/browser owners completed cleanup. Inspect per-case JUnit; original Generic failures/interrupted attempts are not promoted.\n')
