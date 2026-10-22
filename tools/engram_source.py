"""engram_source.py DIR RANK SHA47 SHA48 : DIR/SOURCE.json for a rank's Engram cut, the record the runtime's launcher
checks (rank, world, source revision, each file's size). DIR is the runtime's prepare_engram.py output for RANK: that
rank's 8 of the 24 hash columns of layers 1 and 14, at the original offsets, the other rows left as holes. SHA47 and
SHA48 are the source shards' sha256, which tools/hub_fetch.py checked against the Hub. Standard library only."""
import json, os, sys

d, rank, s47, s48 = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
pinned = {"model-00047-of-00048.safetensors": s47, "model-00048-of-00048.safetensors": s48}
files = {n: {"size": os.stat(f"{d}/{n}").st_size, "source_sha256": pinned[n]}
         for n in sorted(os.listdir(d)) if n.endswith(".safetensors")}
if sorted(files) != sorted(pinned):
    sys.exit(f"{d}: expected the two Engram shards, found {sorted(files)}")
doc = {"rank": rank, "world": 3, "original_repo": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
       "original_revision": "1bae0e1088993b9b82de08f59de5be5a34e42170", "files": files,
       "owned_rows": json.load(open(f"{d}/engram-local.json")), "note": "sparse rank copy; unowned rows are holes"}
with open(f"{d}/SOURCE.json", "x") as f:
    json.dump(doc, f, indent=1)
print(f"rank {rank} SOURCE.json written")
