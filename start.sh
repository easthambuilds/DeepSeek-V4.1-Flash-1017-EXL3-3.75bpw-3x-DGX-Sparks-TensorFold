#!/bin/bash
# start.sh [restart]: set up the three Sparks if needed, start ranks 2 and 1 on the workers and rank 0 (the API) here,
# wait until they load, and send one smoke-test request through all three.
# DRY_RUN=1 ./start.sh: only the preflight (Sparks, cables, routes), then the three rank commands. Changes nothing.
. "$(dirname "$0")/scripts/common.sh"
RESTART=0
[ "${1:-}" = restart ] && RESTART=1

if [ "${DRY_RUN:-0}" = 1 ]; then
  PREFLIGHT_ONLY=1 bash "$ROOT/scripts/prepare.sh"
  if [ -f "$PREPARED" ]; then
    step "The three rank commands (not run)"
    for r in 0 1 2; do
      on $r "python3 -I ${RT[$r]}/launcher/launch.py --config ${RT[$r]}/launcher/cluster.json --variant $ARM --rank $r" \
        | python3 -c "import json,sys; print('   rank $r:', json.load(sys.stdin)['shell_command'][:400], '...')"
    done
  else
    note "not set up yet: ./start.sh runs scripts/prepare.sh first"
  fi
  exit 0
fi

step "Setup"
if [ "$(cat "$PREPARED" 2>/dev/null)" = "$(stamp)" ]; then
  note "ready ($PREPARED)"
else
  if any_running; then
    [ $RESTART = 1 ] || die "a rank is running: ./start.sh restart, or ./stop.sh"
    note "stopping the running ranks before the setup (requests in flight are cut off)"; stop_all
  fi
  bash "$ROOT/scripts/prepare.sh"
fi

step "Checks"
if any_running; then
  [ $RESTART = 1 ] || die "a rank is running: ./start.sh restart, or ./stop.sh"
  note "stopping the running ranks (requests in flight are cut off)"; stop_all
fi
ss -ltn | grep -qE ":($PORT|$MASTER_PORT|$((MASTER_PORT + 1))) " && die "port $PORT, $MASTER_PORT or $((MASTER_PORT + 1)) is in use here"
[ "$HOST" = 127.0.0.1 ] || [ -n "$API_KEY_FILE" ] || note "warning: serving on $HOST with no API key (API_KEY_FILE): anyone who can reach port $PORT can use it"
write_launch_config
for r in 0 1 2; do
  on $r "python3 -I ${RT[$r]}/launcher/launch.py --config ${RT[$r]}/launcher/cluster.json --variant $ARM --rank $r --check > /dev/null" \
    || die "rank $r: the launcher's checks fail (above)"
  # GB10 shares memory between GPU and CPU and reclaims page cache too slowly for CUDA (tools/evict_page_cache.py)
  ev=$(on $r "python3 - ${RT[$r]}/weights ${RT[$r]}/engram ${RT[$r]}/kit" < "$TOOLS/evict_page_cache.py")
  avail=$(on $r "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo")
  note "rank $r: launcher checks pass; $ev; $avail GiB available"
  [ "$avail" -ge "$MIN_AVAIL_GIB" ] || note "warning: rank $r has $avail GiB available (the rank wants ~$MIN_AVAIL_GIB): stop other GPU work"
done

step "Launch"
for r in 2 1 0; do
  on $r "(setsid nohup bash ${RT[$r]}/launcher/run_rank.sh > /dev/null 2>&1 < /dev/null &)"
  note "rank $r started"; sleep 3
done

step "Loading"
t0=$(date +%s)
until [ "$(api /v1/models -o /dev/null -w '%{http_code}')" = 200 ]; do
  for r in 0 1 2; do
    if ! running $r; then
      for q in 0 1 2; do echo "-- rank $q:"; on $q "tail -n 15 ${RT[$q]}/logs/rank$q.log" 2>&1 | cut -c1-220; done
      stop_all
      die "rank $r stopped while loading (logs above; full logs in each Spark's ${RT[0]#$HOME/}/logs/archive)"
    fi
  done
  [ $(( $(date +%s) - t0 )) -lt 1800 ] || { stop_all; die "not up after 30 minutes (logs in each Spark's rt/logs/archive)"; }
  sleep 15
  note "$(( $(date +%s) - t0 )) s: $(tail -n 1 "${RT[0]}/logs/rank0.log" | cut -c1-110)"
done
note "up after $(( $(date +%s) - t0 )) s; $(grep -o 'model [0-9.]* GiB[^"]*left[^"]*' "${RT[0]}/logs/rank0.log" | tail -1 | cut -c1-100)"

step "Smoke test"
reply=$(api /v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\": \"$SERVED_NAME\",
  \"messages\": [{\"role\": \"user\", \"content\": \"What is 17 times 3? Answer with the number only.\"}],
  \"max_tokens\": 2000}")
echo "$reply" | python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; print('   reply:', (m.get('content') or '').strip()[:200])" \
  || die "the smoke test failed: $reply"
echo "$reply" | grep -q '51' || note "warning: the reply does not contain 51"

h=$HOST; [ "$h" = 0.0.0.0 ] && h=$(hostname -I | awk '{print $1}')
step "$SERVED_NAME is live on http://$h:$PORT/v1"
note "logs: ${RT[0]}/logs/rank0.log here, rt/logs/rankN.log on the workers; ./stop.sh stops all three; ./status.sh"
