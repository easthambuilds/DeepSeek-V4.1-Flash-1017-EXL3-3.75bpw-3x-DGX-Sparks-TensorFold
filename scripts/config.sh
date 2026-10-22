# Pins and defaults. Put your own settings in scripts/local.sh (read after this file), not here.

# The checkpoint this recipe serves, and the original weights its Engram tables come from
QUANT_REPO=${QUANT_REPO:-easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.75bpw}
QUANT_REV=${QUANT_REV:-QUANT_REV_PENDING}
BASE_REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
BASE_REV=1bae0e1088993b9b82de08f59de5be5a34e42170
ENGRAM_FILES="model-00047-of-00048.safetensors model-00048-of-00048.safetensors model.safetensors.index.json"
ENGRAM_SHA256_47=76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5
ENGRAM_SHA256_48=9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd

# Runtime: LakyFx's TensorFold TP3 port for three DGX Sparks, at a pinned tag. Its extractor checks the frozen
# TensorFold native runtime snapshot below; Zig builds it for GB10.
RUNTIME_GIT=https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark.git
RUNTIME_TAG=v1.2.2TF
RUNTIME_REV=14a6d49c7cb99b8ce25f9755541da7e7d57cb1d1
RUNTIME_SNAPSHOT=61357debe730ac9ffe20e760edcddbc253b0e525
ZIG_URL=https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz
ZIG_SHA256=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8

# NCCL 2.30.7 with the switchless patches (direct ConnectX-7 cables, no switch): alexellis/switchless-nccl v0.0.1
NCCL_URL=https://github.com/alexellis/switchless-nccl/releases/download/v0.0.1/nccl-2.30.7-switchless-hardened-sm121-linux-arm64.tar.gz
NCCL_TGZ_SHA256=b4a686382a92e57b485ca1bf7cd0f9fde780a68f01ea902ac432b60505b2041f
NCCL_LIB_SHA256=78cb83871792ec57d763d142e4cae26fc754ae284bcc81dcb2a7d50e17d4fa57

# RoPE tables: the runtime's make_rope.py on CPU PyTorch in this image (no network), at its 10M reference pool (it
# checks LakyFx's qualified hashes), then cut to the rows POOL needs. A cut for the 3.3M pool (LakyFx's qualified
# hashes) or the 4M pool (our run of the 3.5 bpw recipe) must match these: rows, table, sha256.
ROPE_IMAGE=${ROPE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}
ROPE_REF_POOL=10000000
ROPE_PINS="3301376 rope-plain-cos.f32=66a47d6f185bdb2962ca0ffe13952ab70409dc15ea52f721a8e09189e5f26fe6
3301376 rope-plain-sin.f32=4ca39e36077fced64beab67b879470deb8f641a1cc26b5d227835c8c09812ecb
3301376 rope-compressed-cos.f32=e02d894c119fc61a26d2aea51488b87d35e4b33f4e266b8335e75aca5155b921
3301376 rope-compressed-sin.f32=d3a39e5b47a1d36ba9c20bfacbdf8803f3df4b7fcf7a0f947e6ef008f0a0aa2b
4001792 rope-plain-cos.f32=8fbb1d468fb70330999c0fe5e94b7300b05abcbb87fa7489db68fa5eca1ac515
4001792 rope-plain-sin.f32=f1b678d1261a576f284b62175b61270be5295401eee6d09c7ad450eb80cae930
4001792 rope-compressed-cos.f32=722ede80087106d4301300fbae5109dd2c8f6f02ae363756c7eaadb7321526b7
4001792 rope-compressed-sin.f32=982c40beb3d55236e6f51de1cdbf2c0f36aa8e8bcabfef4f06991176e5dc91cf"

# The 1017 chat template the launcher requires (LakyFx's qualified one; identical to the checkpoint's)
CHAT_TEMPLATE_SHA256=7d1356675e429ec4a55cbcc1a6fbeb40615dbeb257aca25b8c3705a42087f33e

# Expected outputs of a clean setup (the setup prints them; empty skips the check). Not recorded yet: they are pinned
# after our first end-to-end run of this recipe.
PLAN_SHA256=${PLAN_SHA256-}
RANK_SHA256=${RANK_SHA256-}   # the three rank caches

# Where everything lives on each Spark. Head: ~760 GB at the peak of the setup (checkpoint ~270, Engram source ~203,
# three rank caches ~92 each before ranks 1 and 2 move to their Sparks, Engram cut ~68); ~420 GB after. Others ~165 GB.
DATA_DIR=${DATA_DIR:-$HOME/.cache/dsv41-1017-tp3b}
ARM=TP3B                          # the runtime's name for this checkpoint's rank caches and profile

# The Sparks (see README, Requirements). WORKER1 and WORKER2: ssh targets of ranks 1 and 2, key-based from the head.
WORKER1=${WORKER1:-}
WORKER2=${WORKER2:-}
MASTER_PORT=29660                 # the runtime's rank control ports: 29660 and 29661 on the head

# Server
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8888}
SERVED_NAME=${SERVED_NAME:-deepseek-v4.1-flash}
POOL=${POOL:-3300000}             # tokens of KV shared by all requests (LakyFx's qualified pool size)
CONTEXT=${CONTEXT:-1048576}       # tokens per request (prompt + output); may exceed POOL / PARALLEL
PARALLEL=${PARALLEL:-4}           # concurrent requests; we suggest 2 to 4 with this checkpoint (README, Context)
PROMPT_CACHE_GIB=${PROMPT_CACHE_GIB:-2}
REASONING=${REASONING:-}          # empty: the server's default; or max, high, medium, low, none
API_KEY_FILE=${API_KEY_FILE:-}    # a file holding the API key; every route but /health then needs it
MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-100}   # warn below this much MemAvailable on a Spark before loading
CUDA_LIB=${CUDA_LIB:-/usr/local/cuda/lib64}   # the CUDA runtime libraries the binaries load (DGX OS default)
ALLOW_4K_PAGES=${ALLOW_4K_PAGES:-0}   # we qualified on the 64K-page kernel only
