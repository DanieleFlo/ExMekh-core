"""Remaining actual client qualifications, serial and with recorded x4 deadlines."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path('.').resolve()))
from tools.qualify_metal_command import interrupt_on_termination
from tools.qualify_metal_server import run
root=Path('build/metal-qualification/t06-t07-20261031')
python='agent/.venv/bin/python'

def server(family,name,tests,*,timeout=600,phase=False,vision=False,seed=None,tool_start=None):
    output=root/name
    if (output/'report.json').exists():
        print('EXISTS',name,json.loads((output/'report.json').read_text())['status'],flush=True);return
    print('START',name,flush=True)
    args=argparse.Namespace(family=family,output=output,suite_config=Path('config/native-suite-metal.example.json'),resource_metrics=None,phase_profile=True,kv_seed=seed,vision=vision,mtp=False,phase_server=phase,port=18080,timeout=timeout,timeout_multiplier=4,tool_start_timeout=tool_start,max_footprint_gib=0,reason='Remaining original-model client gates with explicit userrequested4xtimeouts on thisMac. One target-only8192/64 LLM,2GiBSSDcache,2GiBhostreserve; unchanged assertions, public Responses output replay, original native256px image and128px vision preview.',command=tests)
    result=run(args)
    print('FINISHED',name,result['status'],round(result['elapsed_seconds'],2),flush=True)

with interrupt_on_termination():
    acceptance = subprocess.run(['build/metal-venv/bin/python', str(root/'analyze-latest-acceptance.py')], check=False)
    if acceptance.returncode:
        print('DEFER remaining clients: strict image/media numerical acceptance is still open; see product-bf16-media-v3/analysis.json.', flush=True)
        raise SystemExit(0)
    for family in ('flash','qwen27'):
        server(family,f'{family}-phase-cancel-x4',[python,'-m','pytest','-q','-s','tests/test_suite_phase_live.py'],phase=True)
    for family in ('qwen27','flash'):
        server(family,f'{family}-interactive-full-x4',[python,'-m','pytest','-q','-rs','tests/test_interactive_api.py::InteractiveAPI'],timeout=1800)
    server('flash','flash-frontend-file-create-x4',['env','PYTHONPATH=agent/src:agent',python,'-m','pytest','-n','0','--no-cov','-q','-s','agent/tests/test_live_e2e.py::test_live_e2e_interactive_file_creation'],timeout=1200,seed=root/'flash-frontend-file-create-v2/kv')
    server('qwen27','qwen27-exmekh-registered-image-target-only-x4',[python,'-m','pytest','-q','-s','tests/test_suite_exmekh_live.py::ExmekhNativeLive::test_registered_image_tool_retains_conversation'],timeout=1200,seed=root/'qwen27-exmekh-registered-image-v3/kv')
    server('qwen27','qwen27-browser-image-target-only-x4',['env','PYTHONPATH=agent/src:agent','DS4_SUITE_E2E_IMAGE_COUNT=1','DS4_SUITE_E2E_COMPACT_REPLY=1',python,'-m','pytest','-n','0','--no-cov','-q','-s','agent/tests/test_native_image_generation_e2e.py::test_real_image_generation_delivers_pixels_to_model_and_persists_in_ui'],timeout=1800,vision=True,seed=root/'qwen27-browser-image-measured-budget/kv',tool_start=1000)
    (root/'nov01-final-clients-x4-finished.txt').write_text('All cases completed cleanup; inspect each individual verdict and strict quality gate.\n')
