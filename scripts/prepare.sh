#!/bin/bash
# prepare.sh: the one-time setup of all three Sparks, run on the head. Safe to re-run: each step skips what is done.
#   1. preflight and topology    2. runtime (build, NCCL)    3. kit and RoPE tables    4. weights
#   5. plan, receipt, model files    6. Engram, one cut per rank    7. rank caches    8. launcher
# PREFLIGHT_ONLY=1 stops after step 1 (start.sh's DRY_RUN=1).
. "$(dirname "$0")/common.sh"
mkdir -p "$WORK"
B=$DATA_DIR/build          # LakyFx's extracted sources, kit and reference, and the built binaries (head)
LX=$DATA_DIR/src/lakyfx    # LakyFx's repository at RUNTIME_REV (head)
CK=$DATA_DIR/checkpoint    # the checkpoint (head)
ES=$DATA_DIR/engram-src    # DeepSeek's two Engram shards and the 1017 index (head)
K=${RT[0]}/kit
PT=$B/patched/tools
P=$WORK/plan-$ARM.json
filter() { grep -v '^   have' || true; }
for f in "$PREPARED" "$PREPARED.partial"; do
  [ -f "$f" ] || continue
  old=$(cat "$f"); [ "${old% rope=*}" = "$(base_stamp)" ] \
    || die "$DATA_DIR was set up for: ${old% rope=*}. Use a new DATA_DIR for this one, or remove that one."
done
stamp > "$PREPARED.partial"

step "1/8 Preflight and topology"
for r in 1 2; do on $r true || die "no key-based ssh to $(target $r) (ssh-copy-id $(target $r))"; done
for r in 0 1 2; do
  on $r "python3 -" < "$TOOLS/node_facts.py" > "$WORK/facts-r$r.json" || die "rank $r: could not read its facts"
  python3 - "$WORK/facts-r$r.json" "$r" "$ALLOW_4K_PAGES" <<'PY' || exit 1
import json, sys
f, r, allow4k = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3] == "1"
bad = []
if "GB10" not in f["gpu"]: bad.append(f"GPU is '{f['gpu']}', not a GB10")
if f["machine"] != "aarch64": bad.append(f"machine is {f['machine']}")
if f["page_size"] != 65536 and not allow4k:
    bad.append(f"page size {f['page_size']}: we qualified on the 64K-page kernel (linux-nvidia-64k); ALLOW_4K_PAGES=1 to try anyway")
if not f["memlock_unlimited"]: bad.append("memlock is limited (ulimit -l): RDMA needs it unlimited")
if bad: sys.exit("!! rank %s (%s): %s" % (r, f["host"], "; ".join(bad)))
print(f"   rank {r} {f['host']}: {f['gpu']}, kernel {f['kernel']}, {f['mem_avail_kib'] // 1048576} GiB available")
PY
  on $r "command -v rsync >/dev/null && command -v python3 >/dev/null && tar --version | grep -q GNU && command -v sha256sum >/dev/null" \
    || die "rank $r: needs rsync, python3, GNU tar and sha256sum"
  want=$(( r == 0 ? 760 : 175 ))
  free=$(on $r "mkdir -p '${DD[$r]}' && df -BG --output=avail '${DD[$r]}' | tail -1 | tr -dc 0-9")
  [ "$free" -ge "$want" ] || note "warning: rank $r has ${free} GB free under ${DD[$r]} (the setup needs ~$want GB; less if it is partly done)"
done
for c in docker git curl; do command -v $c >/dev/null || die "the head needs $c"; done
docker info >/dev/null 2>&1 || die "the head's Docker is not usable by this user (the RoPE step runs CPU PyTorch in $ROPE_IMAGE)"
python3 "$TOOLS/topology.py" "$WORK"/facts-r{0,1,2}.json "$WORK" || die "the fabric does not fit (above)"
missing=$(python3 - "$WORK/routes.json" <<'PY'
import json, sys
for r in json.load(open(sys.argv[1])):
    if r["dev"]: print(r["rank"], r["to"], r["via"], r["dev"])
PY
)
fix=""
while read -r r to via dev; do
  [ -n "$r" ] || continue
  got=$(on $r "ip -o route get $to" 2>/dev/null || true)
  case "$got" in *"via $via dev $dev"*) ;; *)
    if [ "${APPLY_ROUTES:-0}" = 1 ]; then
      on $r "sudo -n ip route replace $to/32 via $via dev $dev" || die "rank $r: could not add the route (sudo)"
      note "rank $r: added $to/32 via $via dev $dev (until reboot)"
    else fix="$fix\n   rank $r ($(target $r 2>/dev/null)): sudo ip route replace $to/32 via $via dev $dev"; fi ;;
  esac
done <<< "$missing"
[ -z "$fix" ] || die "missing host routes over the cables (README, Requirements); run these, or APPLY_ROUTES=1 for this boot:$(printf "$fix")"
master=$(python3 -c "import json; print(json.load(open('$WORK/topology.json'))['master_address'])")
for r in 1 2; do on $r "ping -c 1 -W 2 $master >/dev/null" || die "rank $r cannot reach the master address $master"; done
note "every rank reaches the master $master over the cables"
[ "${PREFLIGHT_ONLY:-0}" = 1 ] && { rm -f "$PREPARED.partial"; exit 0; }

step "2/8 Runtime: LakyFx $RUNTIME_TAG (${RUNTIME_REV:0:7}), TensorFold snapshot ${RUNTIME_SNAPSHOT:0:8}, NCCL 2.30.7"
if [ "$(git -C "$LX" rev-parse HEAD 2>/dev/null)" != "$RUNTIME_REV" ]; then
  rm -rf "$LX"; mkdir -p "$LX"; git -C "$LX" init -q
  git -C "$LX" fetch -q --depth 1 "$RUNTIME_GIT" "$RUNTIME_REV" && git -C "$LX" checkout -q FETCH_HEAD
  [ "$(git -C "$LX" rev-parse HEAD)" = "$RUNTIME_REV" ] || die "LakyFx's repository is not at $RUNTIME_REV"
fi
if [ ! -x "$B/built/native/bin/tensorfold-native" ] || [ ! -x "$B/built/bin/tf-dsv41-lanes" ]; then
  rm -rf "$B"
  (cd "$LX" && python3 tensorfold/extract.py --out "$B") > "$WORK/extract.log" 2>&1 || die "extract.py: $(tail -3 "$WORK/extract.log")"
  grep -q VERIFIED_EXTRACTED "$WORK/extract.log" && grep -q "$RUNTIME_SNAPSHOT" "$WORK/extract.log" \
    || die "extract.py did not verify runtime $RUNTIME_SNAPSHOT"
  if [ ! -x "$DATA_DIR/zig/zig" ]; then
    curl -fsSL -o "$DATA_DIR/zig.tar.xz" "$ZIG_URL"
    echo "$ZIG_SHA256  $DATA_DIR/zig.tar.xz" | sha256sum -c --quiet - || die "Zig download does not match its sha256"
    mkdir -p "$DATA_DIR/zig" && tar -xJf "$DATA_DIR/zig.tar.xz" -C "$DATA_DIR/zig" --strip-components 1 && rm "$DATA_DIR/zig.tar.xz"
  fi
  note "building the head and follower binaries for sm_121 (about a minute)"
  (cd "$B/source" && ZIG_GLOBAL_CACHE_DIR="$DATA_DIR/zig-cache" "$DATA_DIR/zig/zig" build native dsv41-tools -j8 \
     -Dtarget=aarch64-linux-gnu -Dfatbins="$B/kit/build-fatbins" --prefix "$B/built") > "$WORK/build.log" 2>&1 \
    || die "zig build failed: $(tail -5 "$WORK/build.log")"
fi
if [ ! -f "$DATA_DIR/nccl/.ok" ]; then
  rm -rf "$DATA_DIR/nccl"; mkdir -p "$DATA_DIR/nccl"
  curl -fsSL -o "$DATA_DIR/nccl.tgz" "$NCCL_URL"
  echo "$NCCL_TGZ_SHA256  $DATA_DIR/nccl.tgz" | sha256sum -c --quiet - || die "NCCL download does not match its sha256"
  tar -xzf "$DATA_DIR/nccl.tgz" -C "$DATA_DIR/nccl" && rm "$DATA_DIR/nccl.tgz"
  lib=$(find "$DATA_DIR/nccl" -name libnccl.so.2.30.7 -type f | head -1)
  [ "$(sha256sum < "$lib" | cut -c1-64)" = "$NCCL_LIB_SHA256" ] || die "libnccl.so.2.30.7 does not match its sha256"
  echo "$lib" > "$DATA_DIR/nccl/.ok"
fi
[ -d "$B/patched" ] || python3 -I "$TOOLS/patch_runtime.py" "$LX" "$B/source/tools/dsv41-tp3" "$B/patched" | sed 's/^/   /'
lib=$(cat "$DATA_DIR/nccl/.ok"); libdir=$(dirname "$lib")
mkdir -p "${RT[0]}"/{bin,lib,launcher/profiles,logs,weights/$ARM,weights/models,engram}
cp -p "$B/built/native/bin/tensorfold-native" "$B/built/bin/tf-dsv41-lanes" "$B/built/bin/tf-dsv41-check" "${RT[0]}/bin/"
cp -p "$lib" "${RT[0]}/lib/" && ln -sf libnccl.so.2.30.7 "${RT[0]}/lib/libnccl.so.2" && ln -sf libnccl.so.2 "${RT[0]}/lib/libnccl.so"
cp -p "$libdir"/*.txt "$libdir"/*.md "${RT[0]}/lib/" 2>/dev/null || true
cp -p "$B/patched/launch.py" "${RT[0]}/launcher/launch.py"
for r in 1 2; do
  on $r "mkdir -p ${RT[$r]}/{bin,lib,launcher/profiles,logs,weights/$ARM,weights/models,engram,kit}"
  for d in bin lib; do copy_to $r "${RT[0]}/$d/" "${RT[$r]}/$d/" --delete; done
  copy_to $r "${RT[0]}/launcher/launch.py" "${RT[$r]}/launcher/launch.py"
done
want=$(tree_sum 0 "${RT[0]}/bin")$(tree_sum 0 "${RT[0]}/lib")
for r in 0 1 2; do
  [ "$(tree_sum $r "${RT[$r]}/bin")$(tree_sum $r "${RT[$r]}/lib")" = "$want" ] || die "rank $r: binaries differ from the head's"
  nf=$(on $r "LD_LIBRARY_PATH=${RT[$r]}/lib:$CUDA_LIB ldd ${RT[$r]}/bin/tensorfold-native ${RT[$r]}/bin/tf-dsv41-lanes | grep 'not found'" || true)
  [ -z "$nf" ] || die "rank $r: missing libraries: $nf (CUDA_LIB=$CUDA_LIB)"
done
note "same binaries and NCCL on all three ($want); every library resolves"

step "3/8 Kit and RoPE tables (pool $POOL: $ROPE_ROWS rows)"
REF=$DATA_DIR/rope-ref
if [ ! -f "$REF/rope.json" ]; then
  rm -rf "$REF"; mkdir -p "$REF/models/TF351"
  cp "$B/kit/models/TF351/config.json" "$REF/models/TF351/"   # make_rope.py reads the RoPE settings from the kit's config
  note "computing the $ROPE_REF_POOL-row reference tables on CPU in $ROPE_IMAGE (no network)"
  docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$LX:/lakyfx:ro" -v "$B/reference:/reference:ro" \
    -v "$REF:/kit" "$ROPE_IMAGE" python3 /lakyfx/tensorfold/make_rope.py --reference /reference --kit /kit \
    --pool "$ROPE_REF_POOL" > "$WORK/rope.log" 2>&1 || die "make_rope.py: $(tail -3 "$WORK/rope.log")"
  grep -q CPU_BYTES_MATCH_QUALIFIED_ROPE "$WORK/rope.log" || die "make_rope.py did not reproduce the qualified tables"
fi
if [ "$(python3 -c "import json; print(json.load(open('$K/rope.json'))['rows'])" 2>/dev/null)" != "$ROPE_ROWS" ]; then
  rm -rf "$K"; cp -a "$B/kit" "$K"
  ROPE_PINS=$ROPE_PINS python3 -I "$TOOLS/rope_cut.py" "$REF" "$K" "$POOL" | sed 's/^/   /' \
    || die "RoPE cut failed"
  [ -f "$K/rope.json" ] || die "RoPE cut failed"
fi
note "kit ready ($(du -sh "$K" | cut -f1))"

step "4/8 Weights"
python3 "$TOOLS/hub_fetch.py" "$QUANT_REPO" "$QUANT_REV" "$CK" | filter
python3 "$TOOLS/hub_fetch.py" "$BASE_REPO" "$BASE_REV" "$ES" $ENGRAM_FILES | filter

step "5/8 Checkpoint plan, receipt and model files"
if [ ! -f "$P" ]; then
  python3 -I "$TOOLS/make_plan.py" "$CK" "$ARM" "$QUANT_REPO" "$QUANT_REV" "$LX/tensorfold/preparation/TF29.json" \
    "$LX/tensorfold/preparation" "$P.tmp" > "$WORK/make_plan.log" 2>&1 \
    || die "the checkpoint does not fit the runtime's layout: $(grep -E 'NOT WRITTEN|rror' "$WORK/make_plan.log" | head -3)"
  mv "$P.tmp" "$P"
fi
psha=$(sha256sum < "$P" | cut -c1-64)
[ -z "$PLAN_SHA256" ] || [ "$psha" = "$PLAN_SHA256" ] || die "plan sha256 $psha, pinned $PLAN_SHA256"
note "plan ${psha:0:16}: $(grep -o '"keyed_tensors": [0-9]*' "$WORK/make_plan.log" | tr -d '"'), none missing, no geometry mismatches"
if [ ! -f "$WORK/receipt-$ARM.json" ]; then
  rm -f "$WORK/manifest-$ARM.json"
  python3 -I "$TOOLS/make_arm.py" manifest "$QUANT_REPO" "$QUANT_REV" "$CK" "$WORK/manifest-$ARM.json" | sed 's/^/   /'
  note "re-hashing every shard against the Hub's sha256 (the exporter's input guard)"
  python3 -E -s "$PT/source_verify.py" --manifest "$WORK/manifest-$ARM.json" --root "$CK" --out "$WORK/receipt-$ARM.json.tmp" \
    --weights-only > "$WORK/receipt.log" 2>&1 || die "source_verify.py: $(tail -3 "$WORK/receipt.log")"
  mv "$WORK/receipt-$ARM.json.tmp" "$WORK/receipt-$ARM.json"
fi
[ -f "$K/models/$ARM/config.json" ] || python3 -I "$TOOLS/make_arm.py" kit --tools "$PT" "$CK" "$P" "$K" "$ARM" > "$WORK/kit.log" \
  || die "kit config: $(tail -3 "$WORK/kit.log")"
M=${RT[0]}/weights/models/$ARM
if [ ! -f "$M/model.safetensors.index.json" ]; then
  rm -rf "$M"
  python3 -I "$TOOLS/make_arm.py" models --tools "$PT" "$CK" "$P" "$WORK/receipt-$ARM.json" "$K" "$M" "$ARM" > "$WORK/models.log" 2>&1 \
    || die "model files: $(tail -3 "$WORK/models.log")"
fi
[ "$(sha256sum < "$M/chat_template.jinja" | cut -c1-64)" = "$CHAT_TEMPLATE_SHA256" ] || die "the chat template is not the pinned 1017 one"
for r in 1 2; do
  copy_to $r "$K/" "${RT[$r]}/kit/" --delete
  copy_to $r "$M/" "${RT[$r]}/weights/models/$ARM/" --delete
done
want=$(tree_sum 0 "$K")$(tree_sum 0 "$M")
for r in 1 2; do [ "$(tree_sum $r "${RT[$r]}/kit")$(tree_sum $r "${RT[$r]}/weights/models/$ARM")" = "$want" ] || die "rank $r: kit or model files differ"; done
note "kit and model files identical on all three"

step "6/8 Engram: each rank's 8 of the 24 hash columns (layers 1 and 14), as sparse files"
pids=""
for r in 0 1 2; do
  on $r "test -f ${RT[$r]}/engram/SOURCE.json" && continue
  E=$DATA_DIR/engram-r$r
  [ -f "$E/SOURCE.json" ] && continue
  ( rm -rf "$E"
    python3 -E -s "$LX/tensorfold/prepare_engram.py" --source "$ES" --kit "$K" --rank $r --out "$E" --write > "$WORK/engram-r$r.log" 2>&1 \
      && python3 -I "$TOOLS/engram_source.py" "$E" $r "$ENGRAM_SHA256_47" "$ENGRAM_SHA256_48" >> "$WORK/engram-r$r.log" ) &
  pids="$pids $!"
done
for p in $pids; do wait $p || die "an Engram cut failed (work/engram-r*.log)"; done
for r in 0 1 2; do
  on $r "test -f ${RT[$r]}/engram/SOURCE.json" && continue
  E=$DATA_DIR/engram-r$r
  (cd "$E" && ls | grep -v '\.sha256$' | xargs sha256sum) > "$E.sha256"
  if [ $r = 0 ]; then
    rm -rf "${RT[0]}/engram" && mv "$E" "${RT[0]}/engram"
  else
    note "rank $r: copying its cut over the link (sparse)"
    copy_to $r "$E/" "${RT[$r]}/engram/" --sparse
    on $r "cd ${RT[$r]}/engram && sha256sum -c --quiet" < "$E.sha256" || die "rank $r: Engram copy differs"
    rm -rf "$E"
  fi
  note "rank $r Engram in place and checked"
done

step "7/8 Rank caches (each rank's weights in the runtime's layout, exported on the head)"
p20=$(echo "$psha" | cut -c1-20)
RD=$DATA_DIR/ranks; mkdir -p "$RD"
python3 -I "$TOOLS/make_arm.py" ranks --tools "$PT" "$CK" "$P" "$WORK/receipt-$ARM.json" "$LX/tensorfold/preparation" "$RD" "$ARM" > "$WORK/ranks.cmds"
installed() { on $1 "test -f ${RT[$1]}/weights/$ARM/rank${1}of3-$ARM-$p20.verified.json && test ${RT[$1]}/weights/$ARM/rank${1}of3.bin -ef ${RT[$1]}/weights/$ARM/rank${1}of3-$ARM-$p20.bin"; }
pids=""
for r in 0 1 2; do
  installed $r && continue
  [ -f "$RD/rank${r}of3-$ARM-$p20.verified.json" ] && continue
  ( rm -f "$RD/rank${r}of3-$ARM-$p20".*
    $(sed -n "$((2*r+1))p" "$WORK/ranks.cmds") > "$WORK/rank$r-export.log" 2>&1 \
      && $(sed -n "$((2*r+2))p" "$WORK/ranks.cmds") > "$WORK/rank$r-verify.log" 2>&1 ) &
  pids="$pids $!"
  note "rank $r: exporting (~92 GB) and re-hashing it in full"
done
for p in $pids; do wait $p || die "a rank export or its verification failed (work/rank*-export.log, rank*-verify.log)"; done
i=0
for r in 0 1 2; do
  pin=$(echo "$RANK_SHA256" | awk -v n=$((r+1)) '{print $n}')
  if ! installed $r; then
    v=$RD/rank${r}of3-$ARM-$p20.verified.json
    st=$(python3 -c "import json; d=json.load(open('$v')); print(d['status'], d['full_sha256_checked'])")
    [ "$st" = "CPU_CHECKED_NOT_GPU_QUALIFIED True" ] || die "rank $r verification: $st"
    art=$(python3 -c "import json; print(json.load(open('$RD/rank${r}of3-$ARM-$p20.receipt.json'))['artifact_sha256'])")
    [ -z "$pin" ] || [ "$art" = "$pin" ] || die "rank $r cache sha256 $art, pinned $pin"
    files=$(cd "$RD" && ls rank${r}of3-$ARM-$p20.bin rank${r}of3-$ARM-$p20.receipt.json rank${r}of3-$ARM-$p20.verified.json)
    if [ $r = 0 ]; then
      (cd "$RD" && mv $files "${RT[0]}/weights/$ARM/")
    else
      note "rank $r: copying its cache over the link"
      for f in $files; do copy_to $r "$RD/$f" "${RT[$r]}/weights/$ARM/$f"; done
      got=$(on $r "sha256sum < ${RT[$r]}/weights/$ARM/rank${r}of3-$ARM-$p20.bin" | cut -c1-64)
      [ "$got" = "$art" ] || die "rank $r: the copied cache does not match its receipt"
    fi
    on $r "cd ${RT[$r]}/weights/$ARM && rm -f rank${r}of3.bin && ln rank${r}of3-$ARM-$p20.bin rank${r}of3.bin"
    rm -f "$RD/rank${r}of3-$ARM-$p20".*
  fi
  note "rank $r cache in place: $(on $r "python3 -c \"import json; print(json.load(open('${RT[$r]}/weights/$ARM/rank${r}of3-$ARM-$p20.receipt.json'))['artifact_sha256'][:16])\"")"
done

step "8/8 Launcher"
write_launch_config
for r in 0 1 2; do
  on $r "python3 -I ${RT[$r]}/launcher/launch.py --config ${RT[$r]}/launcher/cluster.json --variant $ARM --rank $r --check > /dev/null" \
    || die "rank $r: the launcher's own checks fail (above)"
done
note "the launcher's checks pass on all three ranks"
mv "$PREPARED.partial" "$PREPARED"
step "Setup ready on all three Sparks"
note "plan $psha"
note "RANK_SHA256=\"$(for r in 0 1 2; do on $r "python3 -c \"import json; print(json.load(open('${RT[$r]}/weights/$ARM/rank${r}of3-$ARM-$p20.receipt.json'))['artifact_sha256'])\""; done | tr '\n' ' ' | sed 's/ $//')\""
