# Shared by start.sh, stop.sh, status.sh and scripts/prepare.sh. Sourced, not run.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
. "$ROOT/scripts/config.sh"
[ -f "$ROOT/scripts/local.sh" ] && . "$ROOT/scripts/local.sh"

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }
die() { printf '\033[31m!! %s\033[0m\n' "$*" >&2; exit 1; }

: "${WORKER1:?set WORKER1=user@host (rank 1) in scripts/local.sh (see scripts/local.sh.example)}"
: "${WORKER2:?set WORKER2=user@host (rank 2) in scripts/local.sh (see scripts/local.sh.example)}"
TOOLS=$ROOT/tools
WORK=$DATA_DIR/work
ROUND_ROWS=32
ROPE_ROWS=$(( (POOL + 49 + 4095) / 4096 * 4096 ))
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)

# on R CMD: run CMD on rank R's Spark (0 = here)
on() {
  local r=$1; shift
  case $r in
    0) bash -c "$*" ;;
    1) "${SSH[@]}" "$WORKER1" "$*" ;;
    2) "${SSH[@]}" "$WORKER2" "$*" ;;
  esac
}
target() { if [ "$1" = 1 ]; then echo "$WORKER1"; else echo "$WORKER2"; fi; }

# Each Spark keeps DATA_DIR at the same place relative to its own home; RT[r] is its runtime root
if [ -z "${DD1:-}" ]; then
  case "$DATA_DIR" in
    "$HOME"/*) DD1="$(on 1 'echo $HOME')${DATA_DIR#"$HOME"}"; DD2="$(on 2 'echo $HOME')${DATA_DIR#"$HOME"}" ;;
    *) DD1=$DATA_DIR DD2=$DATA_DIR ;;
  esac
fi
DD=("$DATA_DIR" "$DD1" "$DD2")
RT=("$DATA_DIR/rt" "$DD1/rt" "$DD2/rt")

# copy_to R SRC DST [rsync options]: bulk copy from here to rank R's Spark, over FABRIC_PEER1/2 when set
copy_to() {
  local r=$1 src=$2 dst=$3; shift 3
  local t peer; t=$(target "$r")
  if [ "$r" = 1 ]; then peer=${FABRIC_PEER1:-}; else peer=${FABRIC_PEER2:-}; fi
  if [ -n "$peer" ]; then case "$t" in *@*) t=${t%%@*}@$peer ;; *) t=$peer ;; esac; fi
  rsync -a --partial "$@" -e "ssh -o BatchMode=yes -c aes128-gcm@openssh.com" "$src" "$t:$dst"
}

# tree_sum R DIR: one digest over every file's sha256 under DIR on rank R (relative paths)
tree_sum() { on "$1" "cd '$2' && find . -type f ! -name '*.partial' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16"; }

# The rank processes and their logs
pattern() { echo "[t]ensorfold-native serve ${RT[$1]}/|[t]f-dsv41-lanes ${RT[$1]}/"; }
running() { on "$1" "pgrep -f '$(pattern "$1")' >/dev/null"; }
any_running() { local r; for r in 0 1 2; do running $r && return 0; done; return 1; }

stop_rank() {  # gzip the rank's log into logs/archive (keeps 10), then stop its process
  local r=$1
  on "$r" "pkill -f '$(pattern "$r")' 2>/dev/null; for i in \$(seq 30); do pgrep -f '$(pattern "$r")' >/dev/null || break; sleep 2; done
    pgrep -f '$(pattern "$r")' >/dev/null && pkill -9 -f '$(pattern "$r")'
    L=${RT[$r]}/logs; mkdir -p \$L/archive
    [ -s \$L/rank$r.log ] && gzip -c \$L/rank$r.log > \$L/archive/\$(date +%Y%m%d-%H%M%S)-rank$r.log.gz && : > \$L/rank$r.log
    ls -1t \$L/archive/*-rank$r.log.gz 2>/dev/null | tail -n +11 | xargs -r rm -f; true"
}
stop_all() { local r; for r in 0 1 2; do stop_rank $r; done; }

api() {  # curl the head's API; $1 path, rest to curl
  local p=$1; shift
  local h=$HOST; [ "$h" = 0.0.0.0 ] && h=127.0.0.1
  local auth=(); [ -n "$API_KEY_FILE" ] && auth=(-H "Authorization: Bearer $(cat "$API_KEY_FILE")")
  curl -s "${auth[@]}" "http://$h:$PORT$p" "$@"
}

# The launcher's files on all three ranks: cluster.json (the topology from the preflight plus the serving settings),
# the profile (pool, context, streams, reasoning, name) and run_rank.sh. Cheap; start.sh rewrites them every start.
write_launch_config() {
  local r
  export QUANT_REPO QUANT_REV CHAT_TEMPLATE_SHA256 POOL CONTEXT PARALLEL ROUND_ROWS PROMPT_CACHE_GIB REASONING SERVED_NAME ROPE_ROWS
  RUNTIME_PROFILE=$DATA_DIR/src/lakyfx/tensorfold/profiles/1.2.0.json python3 -I "$TOOLS/make_arm.py" profile "$WORK/profile-$ARM.json" "$ARM"
  python3 - "$WORK/topology.json" "$WORK/cluster.json" "$HOST" "$PORT" "$MASTER_PORT" "$API_KEY_FILE" "$CUDA_LIB" "${RT[@]}" <<'PY'
import json, sys
topo, out, host, port, mport, key, cuda, *roots = sys.argv[1:]
t = json.load(open(topo))
cfg = {"schema": t["schema"], "master_address": t["master_address"], "master_port": int(mport), "api_host": host,
       "api_port": int(port), "nccl_interface": t["nccl_interface"], "nccl_hcas": t["nccl_hcas"],
       "nccl_address_range": t["nccl_address_range"], "library_path": f"{roots[0]}/lib:{cuda}",
       "ranks": [dict({k: v for k, v in rk.items() if k not in ("host", "addresses")}, root=roots[i],
                      library_path=f"{roots[i]}/lib:{cuda}") for i, rk in enumerate(t["ranks"])]}
if key: cfg["api_key_file"] = key
json.dump(cfg, open(out, "w"), indent=2)
PY
  for r in 0 1 2; do
    printf '%s\n' '#!/bin/bash' \
      "# run_rank.sh: rank $r of DeepSeek-V4.1-Flash-1017 EXL3 3.75 bpw on TensorFold TP3 (written by the recipe's scripts)." \
      "RT=${RT[$r]}; mkdir -p \$RT/logs" \
      "echo \"\$(date '+%F %T') start rank $r\" >> \$RT/logs/rank$r.log" \
      "exec python3 -I \$RT/launcher/launch.py --config \$RT/launcher/cluster.json --variant $ARM --rank $r --run >> \$RT/logs/rank$r.log 2>&1" \
      > "$WORK/run_rank-$r.sh"
    if [ $r = 0 ]; then
      cp "$WORK/cluster.json" "${RT[0]}/launcher/cluster.json"; cp "$WORK/profile-$ARM.json" "${RT[0]}/launcher/profiles/$ARM.json"
      cp "$WORK/run_rank-0.sh" "${RT[0]}/launcher/run_rank.sh"
    else
      copy_to $r "$WORK/cluster.json" "${RT[$r]}/launcher/cluster.json"
      copy_to $r "$WORK/profile-$ARM.json" "${RT[$r]}/launcher/profiles/$ARM.json"
      copy_to $r "$WORK/run_rank-$r.sh" "${RT[$r]}/launcher/run_rank.sh"
    fi
  done
}

# The setup marker: everything that changes the prepared files (not the serving settings)
# (a different POOL only re-cuts the RoPE tables; anything else needs a new DATA_DIR)
base_stamp() { echo "$QUANT_REPO@$QUANT_REV runtime=$RUNTIME_REV nccl=${NCCL_LIB_SHA256:0:12} arm=$ARM roots=${RT[*]}"; }
stamp() { echo "$(base_stamp) rope=$ROPE_ROWS"; }
PREPARED=$DATA_DIR/.prepared
