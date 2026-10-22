"""rope_cut.py REF_KIT KIT POOL : the RoPE tables POOL needs, cut from the runtime's reference tables. Standard library only.

REF_KIT holds the four tables make_rope.py wrote at the 10M reference pool (it checks them against LakyFx's qualified
hashes) and their rope.json. make_rope computes rows position by position in fixed blocks, so the tables for a smaller
pool are exactly the first ROWS rows (ROWS = POOL + 49, rounded up to 4096): truncated to 3,301,376 rows they
reproduce LakyFx's qualified 3.3M hashes. KIT gets the cut tables and their rope.json; a cut whose ROWS has pins in
ROPE_PINS (from scripts/config.sh: lines of ROWS TABLE=SHA256) must match them."""
import hashlib, json, os, sys
from pathlib import Path

ref, kit, pool = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
rows = (pool + 49 + 4095) // 4096 * 4096
r10 = json.loads((ref / "rope.json").read_text())
if r10["rows"] < rows:
    sys.exit(f"the reference tables have {r10['rows']} rows; POOL {pool} needs {rows}")
pins = {}
for l in os.environ.get("ROPE_PINS", "").splitlines():
    if l.strip():
        r, kv = l.split(); n, s = kv.split("="); pins[(int(r), n)] = s
pinned = any(r == rows for r, _ in pins)
out = {"rows": rows, "half": r10["half"]}
for kind in ("plain", "compressed"):
    for part in ("cos", "sin"):
        name = f"rope-{kind}-{part}.f32"
        h = hashlib.sha256()
        with (ref / name).open("rb") as f:
            for b in iter(lambda: f.read(64 << 20), b""):
                h.update(b)
        if h.hexdigest() != r10[name]:
            sys.exit(f"{name}: the reference table changed since make_rope.py wrote it")
        n, h = rows * r10["half"] * 4, hashlib.sha256()
        with (ref / name).open("rb") as src, (kit / name).open("xb") as dst:
            left = n
            while left:
                b = src.read(min(64 << 20, left)); dst.write(b); h.update(b); left -= len(b)
            dst.flush(); os.fsync(dst.fileno())
        out[name] = h.hexdigest()
        if pinned and pins.get((rows, name)) != out[name]:
            sys.exit(f"{name}: the {rows}-row cut does not match the pinned sha256")
(kit / "rope.json").write_text(json.dumps(out, indent=2) + "\n")
print(f"RoPE tables cut to {rows} rows for a {pool}-token pool" + (" (pinned hashes match)" if pinned else ""))
