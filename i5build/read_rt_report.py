"""Extract the readable text of the neural ray tracing report (HTML) for review."""
import html
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

path = r'D:\work\doub\神经光追技术报告.html'
s = open(path, encoding='utf-8', errors='replace').read()
txt = re.sub(r'<script.*?</script>|<style.*?</style>', ' ', s, flags=re.S)
txt = re.sub(r'<[^>]+>', '\n', txt)
txt = html.unescape(txt)
lines = [ln.strip() for ln in txt.split('\n') if ln.strip()]
print('lines:', len(lines))
for ln in lines:
    print(ln)
