"""Replay one pinned HF Gemma block; never instantiate the full model."""
import ctypes as C
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers.models.embedding_gemma2.configuration_embedding_gemma2 import EmbeddingGemma2TextConfig
from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import (
    EmbeddingGemma2EncoderLayer, EmbeddingGemma2RMSNorm,
    EmbeddingGemma2RotaryEmbedding, apply_rotary_pos_emb,
)

sys.path.insert(0,str(Path('tests').resolve()))
from suite_numeric_api import Numeric, View, Attention

torch.set_num_threads(2)
root=Path('build/metal-qualification/t06-t07-20261031')
model=Path('build/metal-oracles/models/google/embeddinggemma-2')
variant=sys.argv[1] if len(sys.argv)>1 else 'gemm-cpu-tail'
layer=int(sys.argv[2]) if len(sys.argv)>2 else 0
config=EmbeddingGemma2TextConfig.from_dict(json.loads((model/'config.json').read_text())['text_config'])
block=EmbeddingGemma2EncoderLayer(config,layer).to(torch.bfloat16).eval()
prefix=f'language_model.layers.{layer}.'
with safe_open(model/'model.safetensors',framework='pt',device='cpu') as sf:
    weights={k[len(prefix):]:sf.get_tensor(k) for k in sf.keys() if k.startswith(prefix)}
block.load_state_dict(weights,strict=True)
def load(stage,width):
    return torch.from_numpy(np.fromfile(root/f'gemma-text/oracle/item-0-{stage}-0.bf16',dtype=np.uint16)).view(torch.bfloat16).reshape(1,-1,width)
def metrics(a,b):
    a=a.bfloat16().reshape_as(b);b=b.bfloat16();d=a.float()-b.float()
    return dict(different=int((a.view(torch.int16)!=b.view(torch.int16)).sum()),elements=a.numel(),max_abs=float(d.abs().max()))
records=[];captured={};handles=[]
with Numeric('metal',library_path=(root/f'neon-model-experiment/{variant}/libnumeric.so').resolve()) as n:
    def hook(name,module,args,output):
        if not args or not isinstance(args[0],torch.Tensor):return
        x=args[0].detach();captured[name]=(x,output.detach() if isinstance(output,torch.Tensor) else output)
        if isinstance(module,torch.nn.Linear):
            w=module.state_dict()['weight'];yv=n.tensor(torch.zeros_like(output));xv=n.tensor(x);wv=n.tensor(w)
            n.linear(yv,xv,wv,x.numel()//x.shape[-1],w.shape[1],w.shape[0])
        elif isinstance(module,EmbeddingGemma2RMSNorm):
            w=module.state_dict().get('weight');yv=n.tensor(torch.zeros_like(output));xv=n.tensor(x)
            wv=n.tensor(w) if w is not None else View()
            n.call('norm',yv,xv,wv,View(),x.shape[-1],config.rms_norm_eps,0,int(w is not None),False)
        elif name.endswith('act_fn'):
            trace=root/f'neon-model-experiment/{variant}'
            x.contiguous().view(torch.uint16).numpy().tofile(trace/(f'layer-{layer}-'+name+'-input.bf16'))
            output.contiguous().view(torch.uint16).numpy().tofile(trace/(f'layer-{layer}-'+name+'-output.bf16'))
            yv=n.tensor(torch.zeros_like(output));n.call('point',yv,n.tensor(x),View(),4,0)
        else:return
        records.append(dict(stage=name,**metrics(n.read(yv,output.shape),output)))
        n.clear()
    for name,module in block.named_modules():
        handles.append(module.register_forward_hook(lambda m,a,o,name=name:hook(name,m,a,o)))
    x=load('embeddings',512) if layer==0 else torch.from_numpy(np.fromfile(root/f'gemma-text/oracle/item-0-layer-{layer-1}.bf16',dtype=np.uint16)).view(torch.bfloat16).reshape(1,-1,512)
    ple=load('ple',12288).reshape(1,x.shape[1],24,512)[:,:,layer]
    rotary=EmbeddingGemma2RotaryEmbedding(config)
    with torch.inference_mode():
        cos,sin=rotary(x,torch.arange(x.shape[1])[None],config.layer_types[layer])
        final=block(x,ple,position_embeddings=(cos,sin))
    for handle in handles:handle.remove()
    oracle=torch.from_numpy(np.fromfile(root/f'gemma-text/oracle/item-0-layer-{layer}.bf16',dtype=np.uint16)).view(torch.bfloat16).reshape_as(final)
    records.append(dict(stage='upstream-block-vs-frozen',**metrics(final,oracle)))
    q=captured['self_attn.q_norm'][1];k=captured['self_attn.k_norm'][1];v=captured['self_attn.v_norm'][1]
    q,k=apply_rotary_pos_emb(q,k,cos,sin)
    seq,heads,dim=x.shape[1],q.shape[1],q.shape[-1]
    q=q.transpose(1,2).contiguous();k=k.transpose(1,2).contiguous();v=v.transpose(1,2).contiguous()
    expected=captured['self_attn.o_proj'][0]
    bounds=np.array([[0,seq]]*seq,dtype=np.uint32)
    parameters=Attention(1,seq,seq,heads,k.shape[2],dim,1,0,True,True)
    yv=n.tensor(torch.zeros_like(q))
    n.call('attend',yv,n.tensor(q),n.tensor(k),n.tensor(v),bounds.ctypes.data_as(C.POINTER(C.c_uint32)),C.byref(parameters))
    records.append(dict(stage='attention-same-input',**metrics(n.read(yv,expected.shape),expected)))
(root/f'neon-model-experiment/{variant}/gemma-layer-{layer}-diagnosis.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
