"""Run the final frontend suite after the prior in-flight collection closes."""
import json,subprocess,time,xml.etree.ElementTree as ET
from pathlib import Path
root=Path('build/metal-qualification/t06-t07-20261031').resolve()
previous=root/'frontend-post-sdk-budget-authorized-junit.xml'
while not previous.exists():time.sleep(1)
final=root/'frontend-post-sdk-budget-final-junit.xml'
if final.exists():raise RuntimeError('Preserve existing final evidence')
with (root/'frontend-post-sdk-budget-final.log').open('w') as log:
 result=subprocess.run(['.venv/bin/python','-m','pytest','-n','2','-q',f'--junitxml={final}'],cwd='agent',stdout=log,stderr=subprocess.STDOUT)
record={'returncode':result.returncode,'command':'agent/.venv/bin/python -m pytest -n 2 -q','scope':'Full frontend after corrected reducible fixture and separate short-answer nonshrinking test; installed SDK97e0119e.'}
if final.exists():
 cases=list(ET.parse(final).iter('testcase'))
 record.update(cases=len(cases),passed=sum(c.find('failure') is None and c.find('error') is None and c.find('skipped') is None for c in cases),skipped=sum(c.find('skipped') is not None for c in cases),failed=sum(c.find('failure') is not None or c.find('error') is not None for c in cases))
record['status']='PASS' if result.returncode==0 else 'FAIL'
(root/'frontend-post-sdk-budget-final-summary.json').write_text(json.dumps(record,indent=2)+'\n')
print(record,flush=True)
raise SystemExit(result.returncode)
