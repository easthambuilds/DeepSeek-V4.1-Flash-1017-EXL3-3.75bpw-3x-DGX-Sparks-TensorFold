#!/bin/bash
# status.sh: each rank's process, memory and GPU, then the API's own /health. Read-only.
. "$(dirname "$0")/scripts/common.sh"
step "Ranks"
for r in 0 1 2; do
  p=$(running $r && echo running || echo "not running")
  m=$(on $r "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo")
  g=$(on $r "nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,clocks.sm --format=csv,noheader" | tr -d ' ')
  note "rank $r $(on $r hostname): $p; $m GiB available; GPU util,temp,SM clock: $g"
done
step "API http://$HOST:$PORT"
api /health | python3 -c "import json,sys; d=json.load(sys.stdin); l=d.get('live',{}); print('   %s: %s, %s connections, %s waiting, decode %.1f tok/s' % (d.get('model'), d.get('status'), l.get('connections'), l.get('waiting'), l.get('decode_tokens_per_second') or 0))" \
  2>/dev/null || note "no answer"
