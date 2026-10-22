# DeepSeek-V4.1-Flash-1017 EXL3 3.75 bpw: TP3 deployment on 3× DGX Spark

Scripts that provision three NVIDIA DGX Spark (GB10) nodes, cabled as a triangle, to serve
[`easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.75bpw`](https://huggingface.co/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.75bpw)
with LakyFx's TensorFold TP3 runtime behind an OpenAI-compatible HTTP API.

- The repository contains no engine code.
- Every external input is pinned by commit or sha256 in `scripts/config.sh`.
- One node (the head, rank 0) runs the scripts and the API. It drives the other two (ranks 1 and 2) over ssh.

**Status: not validated end to end.**
- `PLAN_SHA256` and `RANK_SHA256` are empty, so those two output checks are skipped.
- The scripts are those of the [3.5 bpw recipe](https://github.com/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3x-DGX-Sparks-TensorFold), which passed end to end on 2026-10-21. This repository changes only the checkpoint pins, the RoPE pins and the serving defaults.

## 1. Pinned inputs

| Input | Source | Pin | Size |
| --- | --- | --- | --- |
| Checkpoint | HF `easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.75bpw` | `1285c464297bf2ddd231dba9b7df6a77105273cd` | 270.3 GB, 41 shards |
| Engram source | HF `deepseek-ai/DeepSeek-V4.1-Flash-1017`, shards 47–48 + index | `1bae0e1088993b9b82de08f59de5be5a34e42170`; sha256 `76a05106…` (47), `9dc51433…` (48) | ~203 GB |
| Runtime | `github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark` | tag `v1.2.2TF`, commit `14a6d49c…`; TensorFold snapshot `61357deb…` | |
| Compiler | Zig 0.17.0, aarch64-linux | sha256 `9e8d1166…` | |
| NCCL | `alexellis/switchless-nccl` v0.0.1 (NCCL 2.30.7, switchless patches, sm_121) | archive `b4a68638…`, `libnccl.so.2.30.7` `78cb8387…` | |
| RoPE generator | Docker image `nvcr.io/nvidia/pytorch:26.07-py3` (CPU only, `--network none`) | tag | |
| Chat template | 1017 template (LakyFx-qualified; identical to the checkpoint's) | sha256 `7d135667…` | |

## 2. Node requirements

The preflight (`DRY_RUN=1 ./start.sh`, or step 1 of a full run) checks each item. **Fatal** items stop the run; **Warning** items print a warning and continue.

| Requirement | Value | Check | Enforcement |
| --- | --- | --- | --- |
| Nodes | 3× DGX Spark / GB10, aarch64, 128 GB unified memory | `nvidia-smi -L`, `uname -m` | Fatal |
| Kernel page size | 65536 (`linux-nvidia-64k`) | `getconf PAGESIZE` | Fatal unless `ALLOW_4K_PAGES=1` (4K pages untested) |
| Locked-memory limit | unlimited (RDMA) | `ulimit -l` | Fatal |
| Tools, all nodes | `python3`, `rsync`, GNU `tar`, `sha256sum` | | Fatal |
| Tools, head | `docker` (usable without sudo), `git`, `curl` | `docker info` | Fatal |
| CUDA runtime libraries | `/usr/local/cuda/lib64` (`CUDA_LIB`) | `ldd` on the built binaries | Fatal |
| SSH | key-based, non-interactive, head → `WORKER1`, `WORKER2` | `ssh -o BatchMode=yes <worker> true` | Fatal |
| Free disk under `DATA_DIR` | head ≥ 760 GB during setup (~420 GB after); workers ≥ 175 GB | `df` | Warning |
| `MemAvailable` before load | ≥ 100 GiB per node (`MIN_AVAIL_GIB`) | `/proc/meminfo` | Warning |
| Free ports on head | `PORT` (8888), 29660, 29661 | `ss -ltn` | Fatal |
| GPU clocks | the same cap on all three nodes (ranks synchronise every step). Reference numbers: `nvidia-smi -lgc 0,2000` | | Not checked |

Head disk breakdown at the setup's peak:
- checkpoint ~270 GB;
- Engram source ~203 GB;
- three rank caches of ~92 GB each, until ranks 1 and 2 are moved;
- Engram cuts ~68 GB;
- kit and RoPE ~8 GB.

## 3. Network requirements

| Item | Requirement |
| --- | --- |
| Cabling | 3 direct ConnectX-7 cables, no switch. Each node's two ports go to the two other nodes. |
| Rails | A cabled port appears as two RoCE devices, one per PCIe function (e.g. `rocep1s0f1`, `roceP2p1s0f1`). That gives 4 rails per node, and all 4 must be active. |
| Addressing | Each RoCE device has an IPv4 address. The two devices facing each other on a cable share a subnet used by nothing else: 6 subnets in total. |
| GID | All rails on a node have their IPv4 RoCE v2 GID at the same index. The preflight derives it. |
| Host routes | Three `/32` routes, each over the cable the two nodes share. Rank 2 needs one to the head's address on the head–rank 1 cable. Ranks 1 and 2 each need one to the other's NCCL socket address. |
| Reachability | Ranks 1 and 2 must reach the master address (the head's address on its cable to rank 1). |

- **Routes:** the preflight prints the exact commands, e.g. `sudo ip route replace 192.168.200.2/32 via 192.168.210.2 dev enp1s0f1np1`.
  - `APPLY_ROUTES=1` applies them with `sudo -n` until the next reboot.
  - To keep them across reboots, add them to netplan or NetworkManager.
- **Output:** `tools/topology.py` writes `work/topology.json` and `work/routes.json`. It exits 1 if a node has fewer than 4 rails, a peer has no cable, or a node's rails use mixed GID indexes.

## 4. Configuration

Put overrides in `scripts/local.sh`; start from `scripts/local.sh.example`. It is read after `scripts/config.sh`, which holds the pins and defaults.

| Variable | Default | Description |
| --- | --- | --- |
| `WORKER1`, `WORKER2` | none (required) | ssh targets (`user@host`) of ranks 1 and 2 |
| `DATA_DIR` | `$HOME/.cache/dsv41-1017-tp3b` | Working tree. Placed at the same path relative to `$HOME` on every node. |
| `FABRIC_PEER1`, `FABRIC_PEER2` | unset | Copy bulk data to ranks 1 and 2 over these cable addresses instead of the ssh targets |
| `APPLY_ROUTES` | `0` | `1`: add missing `/32` routes with `sudo -n` (not persistent) |
| `HOST` | `127.0.0.1` | API bind address. A non-loopback address without `API_KEY_FILE` prints a warning. |
| `PORT` | `8888` | API port |
| `API_KEY_FILE` | unset | File holding a bearer token. If set, every route except `/health` requires it. |
| `SERVED_NAME` | `deepseek-v4.1-flash` | Model id in the API |
| `POOL` | `3300000` | KV tokens shared by all requests. RoPE rows = ⌈(POOL + 49) / 4096⌉ × 4096. Hashes are pinned for 3,300,000 (3,301,376 rows, LakyFx's qualified values) and 4,000,000 (4,001,792 rows). Other values are cut without a hash check. |
| `CONTEXT` | `1048576` | Token limit per request (prompt + output). May exceed `POOL / PARALLEL`; requests then compete for the pool. |
| `PARALLEL` | `4` | Concurrent requests |
| `PROMPT_CACHE_GIB` | `2` | Prompt-snapshot memory on the head |
| `REASONING` | server default | `max`, `high`, `medium`, `low` or `none` |
| `MIN_AVAIL_GIB` | `100` | Warning threshold for `MemAvailable` before load |
| `CUDA_LIB` | `/usr/local/cuda/lib64` | CUDA runtime libraries for the binaries |
| `ALLOW_4K_PAGES` | `0` | `1`: skip the page-size check |
| `PLAN_SHA256`, `RANK_SHA256` | empty | Expected plan and rank-cache hashes. Empty skips the check. |

**Defaults:**
- The defaults (`POOL=3300000`, `CONTEXT=1048576`, `PARALLEL=4`; 2 to 4 sessions suggested) are a recommendation derived from the measured memory margins. They are not a measured configuration.
- The measured configuration was `POOL=4000000`, `CONTEXT=401408`, `PARALLEL=12` (section 9).
- On those settings the head had 0.9 GiB left after the runtime's 12.4 GiB reserve.

**When changes take effect:**
- Serving settings (`HOST` through `REASONING`) apply at `./start.sh restart`.
- A new `POOL` re-cuts the RoPE tables on the next start.
- Changing the checkpoint, runtime, NCCL, arm name or node roots invalidates `DATA_DIR`: `prepare.sh` exits and asks for a new one.

## 5. Procedure

Run these on the head:

```bash
git clone https://github.com/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.75bpw-3x-DGX-Sparks-TensorFold.git
cd DeepSeek-V4.1-Flash-1017-EXL3-3.75bpw-3x-DGX-Sparks-TensorFold
cp scripts/local.sh.example scripts/local.sh   # set WORKER1, WORKER2
DRY_RUN=1 ./start.sh                            # preflight only; changes nothing
./start.sh                                      # setup (first run), launch, load, smoke test
```

| Phase | Completion criterion |
| --- | --- |
| `DRY_RUN=1 ./start.sh` | Exit 0. The output lists rails, peer masks, GID index, master address and routes. Once set up, it also prints the three rank commands. |
| `./start.sh` | Exit 0. The smoke-test reply contains `51`. Last line: `== <SERVED_NAME> is live on http://<host>:<PORT>/v1`. |
| First setup | `prepare.sh` prints `plan <sha256>` and `RANK_SHA256="<r0> <r1> <r2>"`. Set these as `PLAN_SHA256` / `RANK_SHA256` in `scripts/local.sh` so later setups are checked against them. |

**Re-running and state:**
- Every setup step checks its outputs and skips completed work, so re-running `./start.sh` after a failure resumes.
- `$DATA_DIR/.prepared.partial` exists while setup is incomplete. `$DATA_DIR/.prepared` holds the stamp of a completed setup.

**Durations** (3.5 bpw recipe, same scripts, 2026-10-21):
- Downloads: ~473 GB, limited by bandwidth.
- Runtime build and RoPE tables: ~2 min.
- Steps 4–8 with the weights already on disk (step 4 only re-checks them against the Hub): 41 min. This checkpoint's rank caches are 6.6% larger: 91.6 GB vs 85.9 GB.
- Model load: 62 s.

## 6. Setup steps (`scripts/prepare.sh`)

| Step | Actions | Checks (failure stops the run) | Outputs |
| --- | --- | --- | --- |
| 1 Preflight, topology | `tools/node_facts.py` on each node; `tools/topology.py` | The section 2 and 3 requirements; workers ping the master address | `work/facts-r*.json`, `work/topology.json`, `work/routes.json` |
| 2 Runtime | Fetch LakyFx at `RUNTIME_REV`. `extract.py`. `zig build native dsv41-tools` for sm_121. Download NCCL. `tools/patch_runtime.py` (section 10). Copy `bin/` and `lib/` to the workers. | Git HEAD equals `RUNTIME_REV`. `extract.py` reports `VERIFIED_EXTRACTED` and the snapshot id. Zig and NCCL sha256 match. Per-node tree digests of `bin/` and `lib/` are equal. `ldd` resolves every library. | `rt/bin`, `rt/lib`, `rt/launcher/launch.py` |
| 3 Kit, RoPE | `make_rope.py` at 10,000,000 rows (CPU, in the RoPE image). `tools/rope_cut.py` to `ROPE_ROWS`. | `CPU_BYTES_MATCH_QUALIFIED_ROPE`. Cut sha256 equals `ROPE_PINS` when the row count is pinned. | `rope-ref/`, `rt/kit` |
| 4 Weights | `tools/hub_fetch.py`: all checkpoint files; Engram shards 47–48 and index. Resumable, 4 files in parallel. | LFS files: sha256. Other files: git blob id. Both against the Hub tree at the pinned revision. | `checkpoint/`, `engram-src/` |
| 5 Plan, receipt, model files | `tools/make_plan.py`; `make_arm.py manifest`; the runtime's `source_verify.py`; `make_arm.py kit` and `models` | All 144,286 keyed tensors resolve, and dtype/shape match the TF29 reference. `PLAN_SHA256` if set. Every shard re-hashed against the Hub. Chat template sha256. Kit and model trees identical on 3 nodes. | `work/plan-TP3B.json`, `work/receipt-TP3B.json`, `rt/weights/models/TP3B` |
| 6 Engram | The runtime's `prepare_engram.py` per rank: 8 of the 24 hash columns of layers 1 and 14, as sparse files. `tools/engram_source.py`. Sparse copy to the workers. | sha256 of every file on each worker | `rt/engram/` with `SOURCE.json` |
| 7 Rank caches | `rank_cache.py --write`, then `rank_verify.py --full-hash`, per rank on the head in parallel; copy to ranks 1 and 2 | Status `CPU_CHECKED_NOT_GPU_QUALIFIED` with `full_sha256_checked: true`. Artifact sha256 equals `RANK_SHA256` if set. Worker copy sha256 equals the receipt. | `rt/weights/TP3B/rank{r}of3.bin`, 91,554,961,832 bytes each |
| 8 Launcher | Write `cluster.json`, profile `TP3B.json` and `run_rank.sh` on all 3 nodes; `launch.py --check` per rank | Launcher check exits 0 on each rank | `$DATA_DIR/.prepared` |

All paths are relative to `DATA_DIR`. `rt/` is the runtime root on every node.

## 7. Operation

| Command | Effect |
| --- | --- |
| `./start.sh` | Set up if needed, then launch ranks 2, 1, 0 (3 s apart), wait up to 30 min for `/v1/models`, and send the smoke test. Refuses if any rank is running. |
| `./start.sh restart` | Stop all ranks (in-flight requests are cut), then start. Applies changed serving settings. |
| `./stop.sh` | Stop all ranks. Each `rankN.log` is gzipped into `rt/logs/archive/` (newest 10 kept). |
| `./status.sh` | Per rank: process, `MemAvailable`, GPU utilisation, temperature and SM clock. Then the API's `/health`. Read-only. |
| `DRY_RUN=1 ./start.sh` | Preflight, plus the three rank commands if set up. No changes. |

**Before load:**
- `start.sh` runs `launch.py --check` on each rank.
- It drops the runtime's own files from the page cache (`tools/evict_page_cache.py`, no privileges needed). GB10 shares memory between CPU and GPU and reclaims page cache too slowly for CUDA allocations.

**If a rank exits during load:** all ranks are stopped, and the last 15 log lines of each rank are printed.

**API:**
- Base URL `http://<HOST>:<PORT>/v1`; model id `SERVED_NAME`.
- Routes: `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/tokenize`, `/health`, `/metrics` (Prometheus).
- Replies carry reasoning in `reasoning_content` before the answer. `max_tokens` covers reasoning plus answer.
- Per-request reasoning effort: `"chat_template_kwargs": {"reasoning_effort": "<level>"}`, or top-level `reasoning_effort`.

**Logs:** `rt/logs/rank{0,1,2}.log` on the respective node.

## 8. Failure reference

Errors print as `!! <message>` and exit non-zero.

| Message (abridged) | Cause | Action |
| --- | --- | --- |
| `no key-based ssh to <worker>` | Missing key | `ssh-copy-id <worker>` |
| `rank r (<host>): GPU is …` / `page size …` / `memlock is limited` | Node outside spec | Section 2 |
| `needs rsync, python3, GNU tar and sha256sum`, `the head needs <tool>` | Missing package | Install it |
| `the head's Docker is not usable by this user` | No Docker access | Add the user to the `docker` group |
| `the fabric does not fit` | Fewer than 4 active rails with IPv4, a missing peer cable, or mixed GID indexes | Section 3; check `ip -4 addr` on every RoCE netdev |
| `missing host routes over the cables` | `/32` routes absent | Run the printed commands, or `APPLY_ROUTES=1` |
| `rank r cannot reach the master address` | Routing or link down | Check the cable between the head and that rank |
| `DATA_DIR was set up for: …` | Pins or node roots changed | Use a new `DATA_DIR`, or remove the old one |
| `… does not match its sha256`, `extract.py did not verify runtime …` | Corrupt or changed download | Delete the file under `DATA_DIR` and re-run |
| `zig build failed` | Build error | `work/build.log` |
| `rank r: binaries differ from the head's`, `missing libraries` | Copy failed, or wrong `CUDA_LIB` | Re-run; set `CUDA_LIB` |
| `make_rope.py did not reproduce the qualified tables` | Different RoPE image or CPU result | `work/rope.log` |
| `<table>: the N-row cut does not match the pinned sha256` | Reference tables corrupted | Remove `rope-ref/` and re-run |
| `the checkpoint does not fit the runtime's layout` | Wrong checkpoint or revision | `work/make_plan.log` |
| `plan sha256 X, pinned Y`, `rank r cache sha256 X, pinned Y` | Output differs from the recorded run | Check the pins and inputs; do not serve |
| `the chat template is not the pinned 1017 one` | Checkpoint template changed | Check `QUANT_REV` |
| `an Engram cut failed`, `rank r: Engram copy differs` | Disk or copy error | `work/engram-r*.log`; re-run |
| `a rank export or its verification failed` | Disk or memory | `work/rank*-export.log`, `work/rank*-verify.log` |
| `rank r: the launcher's checks fail` | Missing or mismatched file on that rank | Launcher output above the message |
| `a rank is running` | Server already up | `./start.sh restart` or `./stop.sh` |
| `port … is in use here` | Port conflict | Free the port or change `PORT` |
| `rank r stopped while loading` | Typically out of memory | Logs printed above. Stop other GPU work; check `MemAvailable`; lower `POOL` or `PROMPT_CACHE_GIB`. |
| `not up after 30 minutes` | Load stalled | `rt/logs/archive/` on each node |
| `the smoke test failed` | API error | The printed reply; `rt/logs/rank0.log` |

## 9. Reference measurements

**Conditions:**
- Our three GB10: Linux 7.0 64K-page kernel, driver 580.178.04.
- The same runtime build, checkpoint weights and runtime changes as this recipe, set up by hand.
- `POOL=4000000`, `CONTEXT=401408`, `PARALLEL=12`, `PROMPT_CACHE_GIB=2`, the server's default reasoning effort.
- GPU clocks locked at 0–2000 MHz on all three; no other load.
- Tools: Bertholomus AI's `kit_bench` (medians, MTP drafts on). Raw data: `provenance/logs/tp3b/` in the model repository.

| Quantity | Value |
| --- | --- |
| Model memory on the head (runtime report) | 97.77 GiB; 0.9 GiB left after the 12.4 GiB reserve |
| Rank cache per node | 91,554,961,832 bytes |
| `MemAvailable` after load, ranks 0 / 1 / 2 | 13.3 / 13.7 / 13.7 GiB |
| `MemAvailable` idle after a 3.4 h benchmark suite | 8.5 / 8.8 / 8.8 GiB |
| Decode, 1 stream, set a: code / prose / structured | 111.9 / 67.5 / 129.6 tok/s |
| Decode, 1 stream, set b (384 tokens): code / prose / structured | 122.6 / 73.8 / 147.6 tok/s |
| Prefill, 8K / 32K / 128K prompt | 2,222 / 2,257 / 2,074 tok/s |
| Aggregate decode, 2 / 4 / 8 streams | 93–102 / 121–136 / 165–181 tok/s |
| GSM8K / MMLU / BFCL v3 (thinking off) | 97.27% / 88.41% / 81.37% |
| Needle, 8K–250K × 3 depths | 12/12 |

The default serving settings (`POOL=3300000`, `CONTEXT=1048576`, `PARALLEL=4`) have not been measured.

## 10. Runtime modifications

`tools/patch_runtime.py` writes modified copies to new files; LakyFx's originals are not edited. Each edit must match its source text exactly once, or the run stops.

1. **Rank-cache exporter and verifier:** one added conversion, BF16 → F32, where the F32 bit pattern is the BF16 bit pattern shifted left 16 bits. This checkpoint stores the Engram k/q projections of layers 1 and 14 as BF16. LakyFx's checkpoint stores them as F32, and the runtime reads F32 either way.
2. **`launch.py`:**
   - Per-rank NCCL socket interface, HCAs and library path (each cable is its own subnet, so there is no shared management interface).
   - Any profile as the variant.
   - Served model name and API key file from the config.
   - The server's own reasoning default when the profile names none.
   - An `operator_trial` profile flag.
   - `--check`, which validates a rank's files without starting it.

## 11. Repository contents

| Path | Function |
| --- | --- |
| `start.sh`, `stop.sh`, `status.sh` | Entry points (section 7) |
| `scripts/config.sh` | Pins and defaults |
| `scripts/local.sh.example` | Template for `scripts/local.sh` (git-ignored) |
| `scripts/common.sh` | Shared functions: ssh, copy, tree digest, process control, launcher config |
| `scripts/prepare.sh` | Setup steps 1–8 |
| `tools/node_facts.py` | Per-node facts as JSON: RoCE devices, IPv4, GIDs, page size, memlock, memory, GPU |
| `tools/topology.py` | Fabric and routes from the three nodes' facts |
| `tools/patch_runtime.py` | The two runtime modifications |
| `tools/rope_cut.py` | Truncates the reference RoPE tables to `ROPE_ROWS` and checks the pins |
| `tools/hub_fetch.py` | Pinned, verified, resumable Hub download (standard library only) |
| `tools/make_plan.py` | Checkpoint plan for the runtime's TP3 layout |
| `tools/make_arm.py` | Manifest, kit config, model files, rank-cache commands, launcher profile |
| `tools/engram_source.py` | `SOURCE.json` for each rank's Engram cut |
| `tools/evict_page_cache.py` | Drops the given files' pages from the page cache |

## 12. Upstream components and licenses

This repository: Apache-2.0 (`LICENSE`). Upstream components keep their own licenses:

- DeepSeek-V4.1-Flash-1017 weights: [DeepSeek](https://huggingface.co/deepseek-ai), MIT.
- TP3 runtime, launcher, exporter, Engram and RoPE tools, kernel kit: [LakyFx](https://github.com/LakyFx), MIT. The checkpoint's per-layer routed-expert widths start from their 1017 3.5 bpw export.
- Native engine: [TensorFold](https://github.com/ashhart/TensorFold) (Ashley Hart and contributors), Apache-2.0. The DeepSeek V4.1 Zig engine family and `kit_bench` are by [Bertholomus AI](https://github.com/bertholomus).
- Format and converter: EXL3 / [exllamav3](https://github.com/turboderp-org/exllamav3) (turboderp), MIT.
- NCCL: [switchless NCCL 2.30.7](https://github.com/alexellis/switchless-nccl) (Alex Ellis / OpenFaaS; [SparkRing](https://github.com/FujitsuPolycom/sparkring) patches) on [NVIDIA NCCL](https://github.com/NVIDIA/nccl).
- Non-expert tensor widths and chat template: [Mia-AiLab](https://huggingface.co/Mia-AiLab).
- Pollard precision map: [bot-lab-21](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard), [WestWaters](https://github.com/WestWaters/pollard-weights).

Not affiliated with or endorsed by DeepSeek, NVIDIA, LakyFx, the TensorFold authors, Bertholomus AI or Mia-AiLab.
