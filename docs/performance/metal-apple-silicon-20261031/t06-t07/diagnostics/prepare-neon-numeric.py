"""Build isolated suite probes; never replace the production binaries."""
import json
import shlex
import subprocess
import sys
from pathlib import Path

root=Path('build/metal-qualification/t06-t07-20261031/neon-model-experiment')
root.mkdir(exist_ok=True)
kernel=r'''
// Fixed FP32 reduction order, independently written from the pinned ATen
// BF16 no-BFDOT operation order. No CPU execution in the native forward.
kernel void nm_bf16_linear(constant ds4_nm_args &a [[buffer(0)]],
                          device uchar *y [[buffer(1)]],
                          device const uchar *x [[buffer(2)]],
                          device const uchar *w [[buffer(3)]],
                          device const uchar *bias [[buffer(4)]],
                          uint gid [[thread_position_in_grid]]) {
    uint rows=a.u[0], inner=a.u[1], columns=a.u[2];
    ulong index=gid;
    if(index>=ulong(rows)*columns) return;
    ulong row=index/columns, col=index%columns;
    float sums[32]={};
    uint aligned=inner/32*32;
    for(uint k=0;k<aligned;k+=32) {
        #pragma unroll
        for(uint lane=0;lane<32;lane++)
            sums[lane]=fma(rd(x,row*inner+k+lane,30),
                           rd(w,col*inner+k+lane,30),sums[lane]);
    }
    for(uint span=16;span>=4;span/=2)
        for(uint lane=0;lane<span;lane++) sums[lane]+=sums[lane+span];
    float result=(sums[0]+sums[1])+(sums[2]+sums[3]);
    float tail[4]={};
    uint aligned8=inner/8*8;
    for(uint k=aligned;k<aligned8;k+=8)
        for(uint lane=0;lane<4;lane++)
            tail[lane]=fma(rd(x,row*inner+k+lane+4,30),rd(w,col*inner+k+lane+4,30),
                           fma(rd(x,row*inner+k+lane,30),rd(w,col*inner+k+lane,30),tail[lane]));
    result+=(tail[0]+tail[1])+(tail[2]+tail[3]);
    for(uint k=aligned8;k<inner;k++)
        result=fma(rd(x,row*inner+k,30),rd(w,col*inner+k,30),result);
    if(a.u[3]) result+=rd(bias,col,a.t[4]);
    wr(y,index,a.t[1],result);
}
'''
header=Path('ds4_numeric_metal_args.h').read_text()
baseline='9805acbbd65a833a5e2ad53dfae49b33616dbd05'
metal=subprocess.check_output(['git','show',baseline+':metal/numeric.metal'],text=True).replace('#include "../ds4_numeric_metal_args.h"','')
objc=subprocess.check_output(['git','show',baseline+':ds4_numeric_metal.m'],text=True)
needle='''        DS4MetalNumeric *s = nm_state(p);
        ds4_num_buffer *xt = NULL, *wt = NULL, *yt = NULL;'''
replacement='''        DS4MetalNumeric *s = nm_state(p);
        if (batch == 1 && !ta && tb && x.type == DS4_NUM_BF16 &&
            w.type == DS4_NUM_BF16 && y.type == DS4_NUM_BF16) {
            ds4_nm_args a = {0};
            a.u[0] = rows; a.u[1] = in; a.u[2] = out; a.u[3] = bias.data != NULL;
            ds4_num_view views[] = {y,x,w,bias};
            return nm_run(p,@"nm_bf16_linear",a,views,4,(uint64_t)rows*out,false);
        }
        ds4_num_buffer *xt = NULL, *wt = NULL, *yt = NULL;'''
assert objc.count(needle)==1
for variant in (sys.argv[1:] or ('gemm','gemm-rms','gemm-cpu-pow')):
    folder=root/variant;folder.mkdir(exist_ok=True)
    profile='gemm-cpu-softmax-sleef' if variant in ('gemm-cpu-compatible','gemm-cpu-layernorm','gemm-cpu-layernorm-stats','gemm-cpu-layernorm-fma','gemm-cpu-layernorm-fma-stats') else variant
    shader=metal+kernel
    if profile.startswith('gemm-cpu-'):
        shader+=(root.parent/'probe-rms-cascade.metal').read_text()
    if profile in ('gemm-cpu-attention','gemm-cpu-precise','gemm-cpu-precise-rsqrt','gemm-cpu-norm-precise','gemm-cpu-tail','gemm-cpu-gelu','gemm-cpu-gelu-precise','gemm-cpu-gelu-unfused','gemm-cpu-gelu-stages','gemm-cpu-gelu-sleef','gemm-cpu-softmax','gemm-cpu-softmax-sleef'):
        start=shader.index('kernel void nm_attention(')
        end=shader.index('kernel void nm_epilogue(',start)
        attention=shader[start:end]
        attention=attention.replace('float maximum = -INFINITY, total = 0, acc0 = 0, acc1 = 0;',
            'float maximum = -INFINITY, total = 0;\n    float acc0[4]={}, acc1[4]={};')
        attention=attention.replace('dot = simd_sum(dot);','''if(a.u[6] && dim%32==0) {
                    dot += simd_shuffle_down(dot,16);
                    dot += simd_shuffle_down(dot,8);
                    dot += simd_shuffle_down(dot,4);
                    dot = (simd_shuffle(dot,0)+simd_shuffle(dot,1))+
                          (simd_shuffle(dot,2)+simd_shuffle(dot,3));
                } else dot=simd_sum(dot);''')
        attention=attention.replace('acc0 += prob[j] *','acc0[(base+j)%4] += prob[j] *').replace('acc1 += prob[j] *','acc1[(base+j)%4] += prob[j] *')
        attention=attention.replace('a.t[1], acc0);','a.t[1], ((acc0[0]+acc0[1])+acc0[2])+acc0[3]);').replace('a.t[1], acc1);','a.t[1], ((acc1[0]+acc1[1])+acc1[2])+acc1[3]);')
        if profile in ('gemm-cpu-tail','gemm-cpu-gelu','gemm-cpu-gelu-precise','gemm-cpu-gelu-unfused','gemm-cpu-gelu-stages','gemm-cpu-gelu-sleef','gemm-cpu-softmax','gemm-cpu-softmax-sleef'):
            attention=attention.replace('[(base+j)%4]','[(base+j)<(keys/4*4) ? (base+j)%4 : 0]')
        shader=shader[:start]+attention+shader[end:]
    if profile=='gemm-cpu-precise-rsqrt':
        shader=shader.replace('pow(sum/width+a.f[0],-0.5f)','1.0f/sqrt(sum/width+a.f[0])')
    if profile in ('gemm-cpu-norm-precise','gemm-cpu-tail','gemm-cpu-gelu','gemm-cpu-gelu-precise','gemm-cpu-gelu-unfused','gemm-cpu-gelu-stages','gemm-cpu-gelu-sleef','gemm-cpu-softmax','gemm-cpu-softmax-sleef'):
        shader=shader.replace('pow(sum/width+a.f[0],-0.5f)','1.0f/precise::sqrt(sum/width+a.f[0])')
    if profile in ('gemm-cpu-gelu','gemm-cpu-gelu-precise','gemm-cpu-gelu-unfused','gemm-cpu-gelu-stages','gemm-cpu-gelu-sleef'):
        old='z = 0.5f * z * (1 + stable_tanh(0.7978845608028654f * (z + 0.044715f * z * z * z)));'
        tanh='precise::tanh' if profile in ('gemm-cpu-gelu-precise','gemm-cpu-gelu-unfused','gemm-cpu-gelu-stages','gemm-cpu-gelu-sleef','gemm-cpu-softmax','gemm-cpu-softmax-sleef') else 'stable_tanh'
        new='{ float cube=z*z*z; z=0.5f*z*(1.0f+'+tanh+'(0.7978845608028654f*(z+0.044715f*cube))); }'
        assert shader.count(old)==1;shader=shader.replace(old,new)
    if profile in ('gemm-cpu-gelu-unfused','gemm-cpu-gelu-stages','gemm-cpu-gelu-sleef'):
        shader='#pragma clang fp contract(off)\n'+shader
    if profile=='gemm-cpu-gelu-sleef':
        shader=(root.parent/'probe-sleef-tanh.metal').read_text()+shader
        shader=shader.replace('precise::tanh(0.7978845608028654f','nm_sleef_tanh(0.7978845608028654f').replace('z = stable_tanh(z);','z = nm_sleef_tanh(z);')
    if profile=='gemm-cpu-gelu-stages':
        shader=shader.replace(new,'{ float cube=z*z*z; z=0.7978845608028654f*(z+0.044715f*cube); }').replace('z = stable_tanh(z);','z = precise::tanh(z);')
    if profile in ('gemm-cpu-softmax','gemm-cpu-softmax-sleef'):
        start=shader.index('kernel void nm_attention(')
        shader=shader[:start]+(root.parent/'probe-attention-softmax.metal').read_text()+'\n'+shader[start:]
        needle_attention='    threadgroup float scores[8], prob[8], stats[2];'
        replacement_attention=needle_attention+'''
    if(a.t[1]==30 && a.t[2]==30 && a.t[3]==30 && a.t[4]==30 &&
       a.u[6] && a.u[7] && !a.u[8] && a.u[5]%32==0) {
        nm_cpu_attention(a,y,q,k,v,bounds,g,tid,scores,prob,stats);
        return;
    }'''
        shader=shader.replace(needle_attention,replacement_attention)
    if profile=='gemm-cpu-softmax-sleef':
        shader=(root.parent/'probe-sleef-tanh.metal').read_text()+shader
        start=shader.index('inline void nm_cpu_attention(')
        end=shader.index('kernel void nm_attention(',start)
        shader=shader[:start]+shader[start:end].replace('exp(', 'nm_sleef_exp(')+shader[end:]
        start=shader.index('kernel void nm_softmax(')
        end=shader.index('kernel void nm_reorder(',start)
        shader=shader[:start]+(root.parent/'probe-softmax-cpu.metal').read_text()+'\n'+shader[end:]
    if profile=='gemm-rms':
        old='float z = (rd(x, ulong(r) * width + j, a.t[2]) - mean) / div;'
        new='float value = rd(x, ulong(r) * width + j, a.t[2]) - mean;\n        float z = kind == 0 ? value * rsqrt(var / width + a.f[0]) : value / div;'
        assert shader.count(old)==1;shader=shader.replace(old,new)
    if variant in ('gemm-cpu-compatible','gemm-cpu-layernorm','gemm-cpu-layernorm-stats','gemm-cpu-layernorm-fma','gemm-cpu-layernorm-fma-stats'):
        old='z = 0.5f * z * (1 + stable_tanh(0.7978845608028654f * (z + 0.044715f * z * z * z)));'
        new='{ float cube=z*z*z; z=0.5f*z*(1.0f+nm_sleef_tanh(0.7978845608028654f*(z+0.044715f*cube))); }'
        assert shader.count(old)==1;shader=shader.replace(old,new)
    if variant in ('gemm-cpu-compatible','gemm-cpu-layernorm','gemm-cpu-layernorm-stats','gemm-cpu-layernorm-fma','gemm-cpu-layernorm-fma-stats'):
        needle_point='    float z = rd(x, i, a.t[2]);'
        special='''    if(a.t[1]==30 && a.t[2]==30 && (a.u[0]==2 || a.u[0]==4)) {
        ushort bits=((device const ushort *)x)[i];
        uint magnitude=bits&0x7fff;
        if(magnitude<0x100) {
            uint halved=magnitude>>1;
            halved+=(magnitude&1)&(halved&1);
            ((device ushort *)y)[i]=ushort((bits&0x8000)|halved);
            return;
        }
    }
'''+needle_point
        assert shader.count(needle_point)==1;shader=shader.replace(needle_point,special)
    inc=folder/'source.inc';inc.write_text('static const char numeric_metal_source[] = '+json.dumps(header+'\n'+shader)+';\n')
    implementation=objc.replace('"build/numeric_metal_source.inc"','"'+str(inc.resolve())+'"').replace('@"nm_conv"','@"nm_conv", @"nm_bf16_linear"',1).replace(needle,replacement)
    if profile.startswith('gemm-cpu-'):
        implementation=implementation.replace('@"nm_bf16_linear"','@"nm_bf16_norm", @"nm_bf16_linear"',1)
        implementation=implementation.replace('return nm_run(p, @"nm_norm", a, v, 4, x.count / width, true);',
            'return nm_run(p, kind == DS4_NUM_RMS && x.type == DS4_NUM_BF16 ? @"nm_bf16_norm" : @"nm_norm", a, v, 4, x.count / width, true);')
    implementation=implementation.replace('DS4MetalNumeric','DS4MetalNumericNeon'+variant.replace('-','').title())
    if profile in ('gemm-cpu-precise','gemm-cpu-precise-rsqrt'):
        implementation=implementation.replace('options.mathMode = MTLMathModeSafe;',
            '{ options.mathMode = MTLMathModeSafe; options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise; }')
    if variant in ('gemm-cpu-layernorm','gemm-cpu-layernorm-stats','gemm-cpu-layernorm-fma','gemm-cpu-layernorm-fma-stats'):
        layernorm_shader=(root.parent/'probe-layernorm-cpu.metal').read_text()
        if variant in ('gemm-cpu-layernorm-stats','gemm-cpu-layernorm-fma-stats'):
            layernorm_shader=layernorm_shader.replace('wr(y,base+j,a.t[1],z);','wr(y,base+j,a.t[1],j%3==0 ? total.mean : (j%3==1 ? total.m2/float(width) : factor));')
        if variant in ('gemm-cpu-layernorm-fma','gemm-cpu-layernorm-fma-stats'):
            layernorm_shader=layernorm_shader.replace('left.mean+c*delta,left.m2+(right.m2+delta*delta*c*float(left.count))', 'fma(c,delta,left.mean),left.m2+fma(delta*delta*c,float(left.count),right.m2)')
        shader+=layernorm_shader
        inc.write_text('static const char numeric_metal_source[] = '+json.dumps(header+'\n'+shader)+';\n')
        implementation=implementation.replace('@"nm_bf16_norm", @"nm_bf16_linear"','@"nm_bf16_layernorm", @"nm_bf16_norm", @"nm_bf16_linear"',1)
        implementation=implementation.replace('return nm_run(p, kind == DS4_NUM_RMS',
            'if(kind == DS4_NUM_LAYER && x.type == DS4_NUM_BF16 && width <= 262144) return nm_run(p,@"nm_bf16_layernorm",a,v,4,x.count/width,true);\n    return nm_run(p, kind == DS4_NUM_RMS')
        if variant in ('gemm-cpu-layernorm-stats','gemm-cpu-layernorm-fma-stats'):
            implementation=implementation.replace('kind == DS4_NUM_LAYER && x.type == DS4_NUM_BF16','kind == DS4_NUM_LAYER && (x.type == DS4_NUM_BF16 || x.type == DS4_NUM_F32)')
    source=folder/'numeric.m';source.write_text(implementation)
    obj=folder/'numeric.o'
    subprocess.run(['clang','-O2','-g','-fPIC','-std=c11','-D_GNU_SOURCE','-Wall','-Wextra','-fno-fast-math','-fobjc-arc','-I.','-c',str(source),'-o',str(obj)],check=True)
    for target in ('tests/probe_qwen_image_encoder','tests/probe_native_suite','tests/probe_gemma_visual'):
        dry=subprocess.check_output(['make','-Bn',target],text=True)
        link=next(line for line in dry.splitlines() if f' -o {target} ' in line)
        args=shlex.split(link)
        args[args.index('-o')+1]=str(folder/Path(target).name)
        args=[str(obj) if a=='ds4_numeric_metal.o' else a for a in args]
        subprocess.run(args,check=True)
    subprocess.run(['clang','-dynamiclib','-o',str(folder/'libnumeric.so'),'ds4_numeric.o',str(obj),'tests/suite_resources.o','-lm','-pthread','-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders'],check=True)
    print('BUILT',variant,flush=True)
