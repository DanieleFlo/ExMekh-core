import json,sys
from pathlib import Path
sys.path.insert(0,str(Path('.').resolve()))
from tools.qualify_metal_command import interrupt_on_termination,run
root=Path('build/metal-qualification/t06-t07-20261031')
with interrupt_on_termination():
 for case in ('reference','rgba'):
  output=root/f'product-encoder-{case}-final-v3';output.mkdir(exist_ok=True)
  fixture=root/f'product-encoder-{case}-final/encoder.ds4i'
  command=['tests/probe_qwen_image_encoder','build/metal-models/image-bf16.ds4b/bundle.ds4b','build/metal-models','build/metal-models/image-tokenizer.ds4q',str(fixture),str(output/'native')]
  result=run(command,output/'run',300,0,'Complete RGB/RGBA encoder after isolated exact position/trig/erf GELU integration and consistent pre-deepstack decoder trace boundaries; unchanged input/oracle/thresholds.',timeout_multiplier=4)
  print(case,result['status'],flush=True)
