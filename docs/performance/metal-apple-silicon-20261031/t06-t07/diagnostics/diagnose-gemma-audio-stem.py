"""Replay only the two Gemma audio subsampling stages on identical CPU inputs."""
import ctypes as C
import json,sys
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioSubSampleConvProjection
sys.path.insert(0,str(Path('tests').resolve()))
from suite_numeric_api import Numeric,View,Conv
root=Path('build/metal-qualification/t06-t07-20261031');oracle=root/'gemma-media/oracle'
variant=sys.argv[1] if len(sys.argv)>1 else 'gemm-cpu-compatible'
model=Path('build/metal-oracles/models/google/embeddinggemma-2')
config=Gemma4AudioConfig.from_dict(json.loads((model/'config.json').read_text())['audio_config'])
torch.set_num_threads(2);module=Gemma4AudioSubSampleConvProjection(config).bfloat16().eval()
prefix='audio_tower.subsample_conv_projection.'
with safe_open(model/'model.safetensors',framework='pt',device='cpu') as sf:
 module.load_state_dict({key[len(prefix):]:sf.get_tensor(key) for key in sf.keys() if key.startswith(prefix)},strict=True)
mask=torch.from_numpy(np.fromfile(oracle/'item-1-media-0-mask.u8',np.uint8)).bool()[None]
x=torch.from_numpy(np.fromfile(oracle/'item-1-media-0-features.f32',np.float32)).reshape(1,mask.shape[1],128).bfloat16()
records=[];handles=[]
def metric(name,a,b):
 a=a.bfloat16().reshape_as(b);b=b.bfloat16();d=a.float()-b.float()
 records.append(dict(stage=name,different=int((a.view(torch.int16)!=b.view(torch.int16)).sum()),elements=a.numel(),max_abs=float(d.abs().max())))
library=(root/f'{variant}/libnumeric.so') if variant.startswith('conv-') else (root/f'neon-model-experiment/{variant}/libnumeric.so')
with Numeric('metal',library_path=library.resolve()) as n:
 def hook(name,m,args,output):
  if not args or not isinstance(args[0],torch.Tensor):return
  x=args[0].detach()
  if isinstance(m,torch.nn.Conv2d):
   weight=m.state_dict()['weight'];y=n.tensor(torch.zeros_like(output))
   c=Conv(x.shape[0],x.shape[1],output.shape[1],m.groups,(1,*x.shape[-2:]),(1,*m.kernel_size),(1,*m.stride),(1,*m.dilation),(0,*m.padding),(0,*m.padding))
   n.call('conv',y,n.tensor(x),n.tensor(weight),View(),C.byref(c))
  elif isinstance(m,torch.nn.LayerNorm):
   weight=m.state_dict()['weight'];y=n.tensor(torch.zeros_like(output))
   n.call('norm',y,n.tensor(x),n.tensor(weight),View(),weight.numel(),m.eps,1,1,False)
  elif isinstance(m,torch.nn.Linear):
   weight=m.state_dict()['weight'];y=n.tensor(torch.zeros_like(output))
   n.call('linear',y,n.tensor(x),n.tensor(weight),View(),x.numel()//x.shape[-1],weight.shape[1],weight.shape[0])
  else:return
  actual=n.read(y,output.shape);metric(name,actual,output)
  trace=(root/variant) if variant.startswith('conv-') else root/f'neon-model-experiment/{variant}'
  for tag,t in (('input',x),('output',output),('weight',m.state_dict()['weight'])):
   t.detach().contiguous().view(torch.uint16).numpy().tofile(trace/f'audio-stem-{name}-{tag}.bf16')
  n.clear()
 for name,m in module.named_modules():handles.append(m.register_forward_hook(lambda m,a,o,name=name:hook(name,m,a,o)))
 with torch.inference_mode():final,_=module(x,mask)
 for handle in handles:handle.remove()
 frozen=torch.from_numpy(np.fromfile(oracle/'clip-0-audio-input-0.bf16',np.uint16)).view(torch.bfloat16).reshape_as(final)
 metric('upstream-stem-vs-frozen',final,frozen)
(library.parent/'gemma-audio-stem-diagnosis.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
