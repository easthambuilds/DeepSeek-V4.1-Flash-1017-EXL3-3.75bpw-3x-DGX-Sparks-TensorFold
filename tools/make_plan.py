"""make_plan.py CKPT ARM REPO REVISION REF_PLAN KEYS_DIR OUT_JSON

Our own checkpoint as a LakyFx TP3 checkpoint plan (schema ds41f.tensorfold.checkpoint-plan.v1), every tensor from
one source (like their TF29 arm, hybrid off). The plan lists both source names because their rank_cache.py takes
--pollard and --mia; both point at CKPT. Checks, before writing:
  - every native key (rank0..2.keys) resolves to a tensor in CKPT (with their wo_a slice.8 -> slice.0 alias);
  - every keyed tensor has the reference plan's (Mia 2.9, TF29) dtype and shape, except a trellis's last axis
    (EXL3 bit width), and an F32 where the reference is F16 only for ffn.gate.bias (their key casts it to f32 anyway);
  - lists tensors present in one but not the other.
Reads safetensors headers only."""
import hashlib, json, re, struct, sys
from pathlib import Path

ckpt, arm, repo, rev, ref_path, keys_dir, out = sys.argv[1:8]
ckpt = Path(ckpt)

def headers(root):
    doc = {}
    for shard in sorted(p.name for p in root.glob("*.safetensors")):
        with open(root / shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
        h.pop("__metadata__", None); doc[shard] = h
    return doc

H = headers(ckpt)
raw = json.dumps({"repo": repo, "revision": rev, "headers": H}, sort_keys=True, separators=(",", ":")).encode()
hsha = hashlib.sha256(raw).hexdigest()
wm = json.loads((ckpt / "model.safetensors.index.json").read_bytes())["weight_map"]
def payload(shard, spec):
    with open(ckpt / shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; lo, hi = spec["data_offsets"]; f.seek(8 + n + lo)
        return hashlib.sha256(f.read(hi - lo)).hexdigest()
tensors, dups = {}, []
for shard, h in H.items():
    for name, spec in h.items():
        lo, hi = spec["data_offsets"]
        t = dict(spec, shard=shard, bytes=hi - lo, source="mia")
        if name in tensors:
            # exllamav3 writes the Engram wkv/k/q projections of layers 1 and 14 into two adjacent shards; the copies
            # must be byte-identical, and the index's shard is the one used.
            u = tensors[name]
            if (u["dtype"], u["shape"]) != (spec["dtype"], spec["shape"]) or payload(u["shard"], u) != payload(shard, spec):
                sys.exit("duplicate tensor differs: " + name)
            dups.append(name)
            if wm[name] != shard: continue
        tensors[name] = t
if any(tensors[n]["shard"] != wm[n] for n in tensors): sys.exit("plan shard differs from the index")

ref = json.loads(Path(ref_path).read_bytes())["tensors"]
keyed = set()
for r in range(3):
    for k in (Path(keys_dir) / f"rank{r}.keys").read_text().splitlines():
        f = k.split("|"); name = f[0]
        if len(f) == 4: name += "." + {"tr": "trellis", "suh": "suh", "svh": "svh"}[f[3]]
        keyed.add(name if name in ref or name in tensors else name.replace(".attn.wo_a.slice.8.", ".attn.wo_a.slice.0."))
alias = {n: (n if n in tensors else n.replace(".attn.wo_a.slice.8.", ".attn.wo_a.slice.0.")) for n in keyed}
missing = sorted(n for n, a in alias.items() if a not in tensors)
bad = []
for n, a in alias.items():
    if a not in tensors or a not in ref: continue
    t, w = tensors[a], ref[a]
    if a.endswith(".trellis"):
        ok = t["shape"][:-1] == w["shape"][:-1] and t["dtype"] == w["dtype"]
    elif a.endswith(".ffn.gate.bias"):
        ok = t["shape"] == w["shape"] and t["dtype"] in ("F16", "F32")
    elif re.search(r"\.engram\.[kq]_weight$", a):
        # the source's BF16 (ours) vs Mia's F32 upcast; the native key casts to float32 (patch_tools.py adds BF16)
        ok = t["shape"] == w["shape"] and t["dtype"] in ("BF16", "F32") and w["dtype"] == "F32"
    else:
        ok = t["shape"] == w["shape"] and t["dtype"] == w["dtype"]
    if not ok: bad.append((a, t["dtype"], t["shape"], w["dtype"], w["shape"]))
only_ours = sorted(set(tensors) - set(ref)); only_ref = sorted(set(ref) - set(tensors))
print(json.dumps({"identical_duplicates": len(dups), "keyed_tensors": len(keyed), "missing_in_ckpt": missing[:20], "n_missing": len(missing),
                  "geometry_mismatch": bad[:20], "n_bad": len(bad),
                  "only_ours": len(only_ours), "only_ours_sample": only_ours[:12],
                  "only_ref": len(only_ref), "only_ref_sample": only_ref[:12]}, indent=1))
if missing or bad: sys.exit("NOT WRITTEN: checkpoint does not fit the native TP3 layout")
src = {"repo": repo, "revision": rev, "headers_sha256": hsha}
doc = {"schema": "ds41f.tensorfold.checkpoint-plan.v1", "arm": arm, "stage": "PRE_PADDING_SOURCE_SELECTION",
       "serving_ready": False, "sources": {"pollard": src, "mia": src},
       "stored_bytes_by_source": {"pollard": 0, "mia": sum(v["bytes"] for v in tensors.values())}, "tensors": tensors}
Path(out).write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
print("WROTE", out, hashlib.sha256(Path(out).read_bytes()).hexdigest())
