# DeepSeek-V4.1-Flash-1017 EXL3 3.75 bpw on 3× DGX Spark with TensorFold

Serve [**DeepSeek-V4.1-Flash-1017**](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) from three NVIDIA DGX
Sparks (GB10, 128 GB each, cabled to each other by their ConnectX-7 ports) through an OpenAI-compatible API, with
**up to 4 concurrent requests of up to 1M tokens** over a **3.3M-token shared KV pool**, images, and DSpark drafting.

This is the recipe of our [3.5 bpw checkpoint](https://github.com/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3x-DGX-Sparks-TensorFold)
for its 3.75 bpw sibling, which has more bits in the routed experts. The scripts are the same; the checkpoint pins, the
RoPE pins and the serving defaults differ.

This repository is a set of scripts and pins, not an engine. The runtime is **LakyFx's TensorFold TP3 port**,
[`deepseek-v41-flash-exl3-tp3-spark`](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark), built from its
pinned sources with its kernels and engine unchanged. It is a three-rank port of Bertholomus AI's DeepSeek Zig engine
for [TensorFold](https://github.com/ashhart/TensorFold). What is ours:

- **The checkpoint:** [`easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.75bpw`](https://huggingface.co/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.75bpw),
  an EXL3 quant of the 1017 weights with agentic calibration, made with published provenance
  ([model card](https://huggingface.co/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.75bpw)).
- **The fit:** a checkpoint plan for the runtime's layout (`tools/make_plan.py`), one exact BF16 → F32 conversion in
  its rank-cache exporter, and a launcher that takes a triangle of direct cables (`tools/patch_runtime.py`).
- **The setup:** `start.sh` / `scripts/prepare.sh` find the cabling, build, download and check every byte, write each
  rank's cache and Engram cut on the head and move them to the other two Sparks, then serve and smoke-test.

| | |
| --- | --- |
| Checkpoint | [`easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.75bpw`](https://huggingface.co/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.75bpw) @ `1285c464`, ~270 GB, plus DeepSeek's original shards 47–48 (Engram tables, ~203 GB) |
| Runtime | [`LakyFx/deepseek-v41-flash-exl3-tp3-spark`](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark) tag `v1.2.2TF` (`14a6d49`), TensorFold runtime snapshot `61357deb`, built with Zig 0.17.0 for sm_121 |
| NCCL | 2.30.7 with the switchless patches, [`alexellis/switchless-nccl`](https://github.com/alexellis/switchless-nccl) `v0.0.1`, sha256-checked |
| RoPE tables | the runtime's `make_rope.py` on CPU in `nvcr.io/nvidia/pytorch:26.07-py3`, cut to the pool |
| API | `http://127.0.0.1:8888/v1`, model id `deepseek-v4.1-flash` |
| Defaults | pool 3,300,000 tokens, 1,048,576 tokens per request, 4 streams, 2 GiB prompt cache, the server's default reasoning effort |

**Status:** not yet run end to end. The plan and rank-cache hashes a clean setup must produce are pinned after our first
end-to-end run (`PLAN_SHA256` and `RANK_SHA256` in `scripts/config.sh` are empty until then, which skips those checks).

## Results

Measured on our three GB10 (Linux 7.0 64K-page kernel, driver 580.178.04) with the same runtime build, checkpoint
weights and runtime changes that this recipe sets up, which we set up by hand, **but with the 3.5 bpw recipe's serving
settings**: a 4M-token pool, 401,408 tokens per request, 12 streams. **GPU clocks capped at 2,000 MHz on all three** for
every number here. Raw results are in the model's `provenance/logs/tp3b/`.

- **Decode, one stream** (Bertholomus AI's `kit_bench`, medians, MTP drafts on): 111.9 tok/s writing code and 67.5
  writing prose (set a). Our 3.5 bpw quant on the same runtime and cap: 109.2 and 71.2.
- **Concurrent streams, total:** 2 streams 93–102, 4 streams 121–136, 8 streams 165–181 tok/s.
- **Prefill:** 2,222 / 2,257 / 2,074 tok/s at 8K / 32K / 128K-token prompts.
- **Trade-off:** against our 3.5 bpw quant, decode is 5% slower on prose and 14% slower on short structured text, code
  is level, and 2 and 4 streams are 5–9% slower. KL against DeepSeek's FP8 release is 4–20% lower on all five test
  sets ([model card](https://huggingface.co/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.75bpw)).
- **Benchmarks:** GSM8K 97.3, MMLU 88.4, BFCL v3 81.4 (thinking off), needle 12/12 at 8K–250K. Level with our
  3.5 bpw quant; no paired difference is significant.
- **Memory:** the model takes 97.8 GiB on the head (3.5 bpw: 92.5). With the 4M pool allocated and idle, the three
  Sparks have 13.3 / 13.7 / 13.7 GiB available, and 8.5 / 8.8 / 8.8 GiB at idle after a 3.4-hour benchmark suite.
  Hence this recipe's smaller defaults (below, Context).

## Requirements

- **Three DGX Sparks** (or GB10 systems with 128 GB unified memory) with nothing else large on their GPUs. Each
  wants ~100 GiB available when its rank starts; `start.sh` first drops the runtime's own files from the page cache
  (`tools/evict_page_cache.py`): GB10 reclaims page cache too slowly for CUDA.
- **Cabled as a triangle:** each Spark's two ConnectX-7 ports go to the other two, three cables in all, no switch.
  Every cable shows up as two RoCE devices per end (one per PCIe function, e.g. `rocep1s0f1` / `roceP2p1s0f1`), and
  each of those needs an IPv4 address, in a subnet of its own shared only with the device at the cable's other end:
  six subnets in all. The preflight works out which port reaches which Spark, the RoCE v2 GID index and the rest
  from these addresses (`DRY_RUN=1 ./start.sh` shows it).
- **Host routes:** with one subnet per cable, the third Spark cannot reach the head's cable to the second, and the
  two workers cannot reach each other's NCCL socket address. The preflight lists the three /32 routes this needs,
  each over the cable the two Sparks share, e.g. `sudo ip route replace 192.168.200.2/32 via 192.168.210.2 dev
  enp1s0f1np1`. `APPLY_ROUTES=1` adds them with sudo until the next reboot; make them permanent in your netplan or
  NetworkManager profiles.
- **The 64K-page kernel** (`linux-nvidia-64k`, `getconf PAGESIZE` = 65536) on all three: we measured on it only.
  `ALLOW_4K_PAGES=1` skips the check. Memlock unlimited (`ulimit -l`) for RDMA.
- **Key-based ssh** from the head (rank 0, which runs the API) to the other two (`WORKER1`, `WORKER2`).
- `python3`, `rsync` and GNU `tar` on all three; Docker (for the RoPE step only), `git` and `curl` on the head. The
  CUDA runtime libraries in `/usr/local/cuda/lib64` (`CUDA_LIB`), as on DGX OS.
- **Disk:** on the head, ~760 GB under `DATA_DIR` at the peak of the setup (checkpoint ~270 GB, Engram source
  ~203 GB, three rank caches ~92 GB each before two of them move, Engram cut ~68 GB, kit and RoPE ~8 GB); ~420 GB
  after. On the other two, ~165 GB each. (Scaled from the 3.5 bpw recipe's run by the larger checkpoint and caches.)
- **Same GPU clock cap on all three** if you compare numbers: the ranks wait for each other every step, so the
  slowest clock sets the pace. We use `sudo nvidia-smi -lgc 0,2000`.

## Quick start

On the head:

```bash
git clone https://github.com/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.75bpw-3x-DGX-Sparks-TensorFold.git
cd DeepSeek-V4.1-Flash-1017-EXL3-3.75bpw-3x-DGX-Sparks-TensorFold
cp scripts/local.sh.example scripts/local.sh   # WORKER1, WORKER2
DRY_RUN=1 ./start.sh                            # checks the Sparks, cables and routes; changes nothing
./start.sh
```

The first run sets up all three Sparks (below), which takes hours, mostly downloading. Later starts only load
(a few minutes). `./start.sh` ends with one request through all three ranks and prints the endpoint:

```bash
curl -s http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4.1-flash",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 4000
}'

./start.sh restart        # restart all three ranks (picks up changed settings)
./stop.sh                 # stop all three; each rank's log is kept in rt/logs/archive
./status.sh               # processes, memory, GPUs and the API's /health
```

The model thinks before it answers (`reasoning_content`), so give replies enough `max_tokens`. The server listens on
`127.0.0.1` by default. Set `HOST=0.0.0.0` in `scripts/local.sh`, together with `API_KEY_FILE`, to serve the
network. `GET /metrics` gives Prometheus counters; `GET /health` a live view.

## What `scripts/prepare.sh` does

Run on the head; it drives the other two over ssh. Each step skips work already done, so it is safe to re-run.

1. **Preflight and topology:** the GPU, page size, memlock, tools and disk on each Spark; the cables
   (`tools/node_facts.py`, `tools/topology.py`): four rails per Spark, which rails reach which peer, the GID index,
   the master address, the NCCL address range; the host routes; the workers reach the master over the cables.
2. **Runtime:** LakyFx's repository at the pinned commit. Its `extract.py` checks and unpacks the frozen TensorFold
   sources and kernel kit. Zig 0.17.0 (sha256-checked) builds the head and follower binaries. NCCL 2.30.7 switchless
   (sha256-checked, archive and library). `tools/patch_runtime.py` writes the patched exporter and launcher (below).
   Binaries and NCCL are copied to the other two and compared, and `ldd` must resolve every library.
3. **Kit and RoPE tables:** the runtime's `make_rope.py` at its 10M reference pool, on CPU without network; it
   checks LakyFx's qualified hashes. `tools/rope_cut.py` keeps the rows the pool needs (for the 3.3M pool, the first
   3,301,376 rows, which must match LakyFx's qualified hashes; for a 4M pool, 4,001,792 rows, pinned from our run).
4. **Weights:** `tools/hub_fetch.py` downloads the checkpoint and DeepSeek's shards 47–48 at pinned revisions and
   checks every file against the Hub: sha256 for large files, the git blob id for small ones.
5. **Plan, receipt, model files:** `tools/make_plan.py` maps every tensor the runtime loads to a shard and byte range
   of the checkpoint and checks it against the runtime's reference geometry (144,286 keyed tensors, none missing). The
   runtime's `source_verify.py` re-hashes every shard against the Hub's sha256. `tools/make_arm.py` writes the
   runtime's config and the model folder (tokenizer, chat template, the 266 vision tensors compared byte for byte).
6. **Engram:** the runtime's `prepare_engram.py` cuts each rank's 8 of the 24 hash columns (layers 1 and 14) from
   DeepSeek's shards as sparse files; ranks 1 and 2 get theirs over the link, checked by sha256 there.
7. **Rank caches:** the runtime's `rank_cache.py` writes each rank's weights (~92 GB) on the head, and `rank_verify.py`
   re-reads them in full. Ranks 1 and 2 get theirs over the link; each must match its exporter receipt.
8. **Launcher:** the cluster config (from step 1), the profile and `run_rank.sh` on all three, then the launcher's own
   checks for each rank.

**The two runtime changes,** both written to new files by `tools/patch_runtime.py`, which stops if any edit does not
match its pinned source exactly once:

- The exporter and its verifier gain one exact conversion, BF16 → F32 (the F32 bit pattern is the BF16 one shifted up
  16 bits). This checkpoint keeps the source's BF16 Engram k/q projections of layers 1 and 14; LakyFx's own checkpoint
  stores them as F32, and the runtime reads them as F32 either way.
- The launcher takes per-rank NCCL socket interfaces, HCAs and library paths (a triangle with a subnet per cable has
  no shared management interface), any profile, the served name and an API key file, and `--check`.

## Configuration

`scripts/config.sh` holds the pins and defaults. Override them in `scripts/local.sh`: `WORKER1`, `WORKER2`,
`DATA_DIR`, `HOST`, `PORT`, `API_KEY_FILE`, `SERVED_NAME`, `POOL`, `CONTEXT`, `PARALLEL`, `PROMPT_CACHE_GIB`,
`REASONING`, `FABRIC_PEER1` / `FABRIC_PEER2` (copy the setup's files over these cable addresses), `APPLY_ROUTES`,
`MIN_AVAIL_GIB`, `CUDA_LIB` and `ALLOW_4K_PAGES`. Serving settings take effect at `./start.sh restart`; a larger
`POOL` re-cuts the RoPE tables first.

**Context:** `CONTEXT` is each request's window and `POOL` the KV all requests share. A request's window may exceed
`POOL / PARALLEL`; requests then compete for the pool. The runtime and its qualification use 401,408 tokens per request.

The defaults are our recommendation for this checkpoint, **not a measured configuration**: a 3.3M-token pool
(LakyFx's qualified pool size), 1M tokens per request, and 2 to 4 concurrent sessions (`PARALLEL=4`; set 2 for more
headroom). The checkpoint leaves about 4.5 GiB less memory available on each Spark than our 3.5 bpw quant, whose recipe
defaults to a 4M pool and 12 streams. Every number under Results was measured with those larger settings.

## Credits

- **[DeepSeek](https://huggingface.co/deepseek-ai):** DeepSeek-V4.1-Flash-1017 (MIT).
- **[LakyFx](https://github.com/LakyFx):** the TensorFold TP3 runtime for three DGX Sparks (MIT): its launcher,
  exporter, Engram and RoPE tools, profiles and kernel kit, used as published apart from the two changes above; and the
  per-layer routed-expert widths of their 1017 3.5 bpw export, the starting point of the checkpoint's recipe.
- **[Ashley Hart](https://github.com/ashhart) and the [TensorFold](https://github.com/ashhart/TensorFold)
  contributors:** the native Zig CUDA engine. **[Bertholomus AI](https://github.com/bertholomus):** the DeepSeek
  V4.1 Zig engine family the TP3 port builds on, and `kit_bench`.
- **[Mia-AiLab](https://huggingface.co/Mia-AiLab):** the 2.9 bpw per-tensor map the checkpoint's non-expert tensors
  follow, the chat template, and the conventions of their DGX Spark recipes.
- **Pollard precision map:** [bot-lab-21](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard)
  and [WestWaters](https://github.com/WestWaters/pollard-weights).
- **[turboderp](https://github.com/turboderp-org/exllamav3):** ExLlamaV3 and the EXL3 format.
- **[Alex Ellis / OpenFaaS](https://github.com/alexellis/switchless-nccl)** and
  **[SparkRing](https://github.com/FujitsuPolycom/sparkring):** the switchless NCCL 2.30.7 build and patches, on
  [NVIDIA NCCL](https://github.com/NVIDIA/nccl).

Not affiliated with or endorsed by DeepSeek, NVIDIA, LakyFx, the TensorFold authors, Bertholomus AI or Mia-AiLab.

## License

Apache-2.0 for the scripts and tools in this repository (`LICENSE`). The checkpoint, runtime, NCCL build and
DeepSeek's weights keep their own licenses.
