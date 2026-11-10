import json,subprocess
from pathlib import Path
root=Path('build/metal-qualification/t06-t07-20261031');folder=root/'neon-erf';folder.mkdir(exist_ok=True)
s=Path('build/numeric_metal_source.inc').read_text();shader=json.loads(s[len('static const char numeric_metal_source[] = '):-2]);needle='// erfc approximation';at=shader.index(needle);shader=shader[:at]+(root/'probe-neon-erf.metal').read_text()+'\n'+shader[at:];shader=shader.replace('z = nm_gelu(z);','z = a.t[1]==30 && a.t[2]==30 ? nm_bf16_gelu_erf(z) : nm_gelu(z);');inc=folder/'source.inc';inc.write_text('static const char numeric_metal_source[] = '+json.dumps(shader)+';\n')
s=Path('ds4_numeric_metal.m').read_text().replace('DS4MetalNumeric','DS4MetalNumericNeonErf').replace('"build/numeric_metal_source.inc"','"'+str(inc.resolve())+'"');p=folder/'numeric.m';p.write_text(s);obj=folder/'numeric.o'
subprocess.run(['clang','-O2','-fPIC','-std=c11','-D_GNU_SOURCE','-Wall','-Wextra','-fno-fast-math','-fobjc-arc','-I.','-c',str(p),'-o',str(obj)],check=True)
subprocess.run(['clang','-dynamiclib','-o',str(folder/'libnumeric.so'),'ds4_numeric.o',str(obj),'tests/suite_resources.o','-lm','-pthread','-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders'],check=True)
