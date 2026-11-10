import json,subprocess
from pathlib import Path
root=Path('build/metal-qualification/t06-t07-20261031');folder=root/'audio-channel-cascade-v2';folder.mkdir(exist_ok=True)
s=Path('build/numeric_metal_source.inc').read_text();shader=json.loads(s[len('static const char numeric_metal_source[] = '):-2])
shader+=chr(10)+(root/'probe-channel-rms-cpu.metal').read_text()
old='float z = rd(x, ulong(c) * spatial + p, a.t[2]) / div;'
new='float z = a.u[2] && a.t[2]==30 ? rd(x,ulong(c)*spatial+p,a.t[2]) * (1.0f/precise::sqrt(sum/channels+a.f[0])) : rd(x, ulong(c) * spatial + p, a.t[2]) / div;'
assert shader.count(old)==1;shader=shader.replace(old,new)
inc=folder/'source.inc';inc.write_text('static const char numeric_metal_source[] = '+json.dumps(shader)+';\n')
s=Path('ds4_numeric_metal.m').read_text().replace('DS4MetalNumeric','DS4MetalNumericAudioChannelCascadeV2').replace('"build/numeric_metal_source.inc"','"'+str(inc.resolve())+'"')
s=s.replace('@"nm_channel",', '@"nm_channel", @"nm_bf16_channel_rms",',1).replace('return nm_run(p, @"nm_channel",','return nm_run(p, rms && x.type==DS4_NUM_BF16 ? @"nm_bf16_channel_rms" : @"nm_channel",')
p=folder/'numeric.m';p.write_text(s);obj=folder/'numeric.o'
subprocess.run(['clang','-O2','-fPIC','-std=c11','-D_GNU_SOURCE','-Wall','-Wextra','-fno-fast-math','-fobjc-arc','-I.','-c',str(p),'-o',str(obj)],check=True)
subprocess.run(['clang','-dynamiclib','-o',str(folder/'libnumeric.so'),'ds4_numeric.o',str(obj),'tests/suite_resources.o','-lm','-pthread','-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders'],check=True)
print('BUILT audio channel precise')
