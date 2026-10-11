#!/bin/bash
# Is the new release visible on the index this machine actually uses?
echo "--- pip config ---"
python3 -m pip config list 2>&1 | head -5
for f in /etc/pip.conf "$HOME/.config/pip/pip.conf" "$HOME/.pip/pip.conf"; do
    [ -f "$f" ] && { echo "== $f"; cat "$f"; }
done
echo
echo "--- pypi.org simple index ---"
curl -s --max-time 90 https://pypi.org/simple/bitsandbytes-cpu-fork/ \
    | grep -o 'bitsandbytes_cpu_fork-[0-9a-zA-Z._-]*' | sort -u
echo
echo "--- json api (files per version) ---"
curl -s --max-time 90 https://pypi.org/pypi/bitsandbytes-cpu-fork/json \
    | python3 -c "
import json,sys
try:
    d = json.load(sys.stdin)
except Exception as e:
    print('  json unavailable:', e); raise SystemExit
for v in sorted(d['releases']):
    for f in d['releases'][v]:
        print(' ', v, f['filename'], f['size'])
print('  info.version =', d['info']['version'])
print('  requires_dist =', d['info'].get('requires_dist'))
"
