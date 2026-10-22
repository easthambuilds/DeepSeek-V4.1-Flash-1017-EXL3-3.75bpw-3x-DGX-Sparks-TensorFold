"""patch_runtime.py LAKYFX_REPO EXTRACTED_TOOLS OUT_DIR : the recipe's two small changes to LakyFx's runtime, written as
new files under OUT_DIR (the originals are not touched). Every replacement must match exactly once, so a different
runtime version stops here instead of being patched blindly. Standard library only.

  OUT_DIR/tools/      a copy of the extracted tools/dsv41-tp3 (the rank-cache exporter and its verifier) with one exact
                      conversion added, BF16 -> float32: the float32 bit pattern is the bf16 one shifted up 16 bits.
                      This checkpoint keeps the source's BF16 Engram k/q projections (layers 1 and 14); LakyFx's own
                      checkpoint stores them upcast to F32, and the runtime's keys read float32 either way.
  OUT_DIR/launch.py   LakyFx's tensorfold/launch.py with per-rank settings for a triangle of direct cables (each cable
                      its own subnet, so there is no shared management interface: per-rank NCCL socket interface, HCAs
                      and library path), any profile in profiles/ as the variant, the served model name and an API key
                      file from the config, the server's own reasoning default when the profile names none, an
                      operator_trial profile flag, and --check (validate a rank's files without starting it)."""
import shutil, struct, sys
from pathlib import Path

repo, tools, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)


def edit(path, edits):
    text = path.read_text()
    for old, new in edits:
        if text.count(old) != 1:
            sys.exit(f"{path.name}: expected exactly one match for {old[:70]!r}")
        text = text.replace(old, new)
    path.write_text(text)


shutil.copytree(tools, out / "tools")   # fails if it exists
edit(out / "tools/rank_cache.py", [
    ("if dtype!=DTYPE[t['dtype']] and not (t['dtype']=='F16' and dtype=='float32'):",
     "if dtype!=DTYPE[t['dtype']] and not (t['dtype'] in ('F16','BF16') and dtype=='float32'):"),
    ("                data=b''.join(struct.pack('<f',x[0]) for x in struct.iter_unpack('<e',data))\n",
     "                data=b''.join(struct.pack('<f',x[0]) for x in struct.iter_unpack('<e',data))\n"
     "            elif l['spec']['dtype']=='BF16' and l['dtype']=='float32':\n"
     "                data=b''.join(b'\\x00\\x00'+data[i:i+2] for i in range(0,len(data),2))\n")])
edit(out / "tools/rank_verify.py", [
    ("        value=struct.pack('<f',struct.unpack('<e',value)[0])\n",
     "        value=struct.pack('<f',struct.unpack('<e',value)[0])\n"
     "    elif spec['dtype']=='BF16' and info['dtype']=='float32':\n"
     "        value=b'\\x00\\x00'+value\n")])
for bits in (0x3F80, 0xBF80, 0x0000, 0x8000, 0x7F7F, 0x0001, 0x4049, 0xC2F7, 0x7F80, 0x3E4C):
    f = struct.unpack("<f", b"\x00\x00" + struct.pack("<H", bits))[0]
    assert struct.unpack("<I", struct.pack("<f", f))[0] == bits << 16

shutil.copyfile(repo / "tensorfold/launch.py", out / "launch.py")
edit(out / "launch.py", [
    ('"""Print or run one explicitly configured TP3 rank. Does not replace existing services."""',
     '"""Print or run one explicitly configured TP3 rank. Does not replace existing services.\n\n'
     'Patched by DeepSeek-V4.1-Flash-1017-EXL3-3x-DGX-Sparks-TensorFold (tools/patch_runtime.py): per-rank\n'
     'nccl_interface, nccl_hcas and library_path, any profile as the variant, served_name and api_key_file, an optional\n'
     'reasoning effort, the operator_trial flag, and --check."""'),
    ("'NCCL_IB_HCA': ','.join(config['nccl_hcas'])", "'NCCL_IB_HCA': ','.join(node.get('nccl_hcas', config['nccl_hcas']))"),
    ("'NCCL_SOCKET_IFNAME': '=' + config['nccl_interface']",
     "'NCCL_SOCKET_IFNAME': '=' + node.get('nccl_interface', config['nccl_interface'])"),
    ("    if config.get('library_path'):\n        env['LD_LIBRARY_PATH'] = config['library_path']",
     "    if node.get('library_path', config.get('library_path')):\n"
     "        env['LD_LIBRARY_PATH'] = node.get('library_path', config.get('library_path'))"),
    ("'--name', 'deepseek-v4.1-flash',", "'--name', profile.get('served_name', 'deepseek-v4.1-flash'),"),
    ("'--reasoning-effort', profile['reasoning'], ",
     "*(['--reasoning-effort', profile['reasoning']] if profile.get('reasoning') else []), "),
    ("        if benchmark:\n            argv.append('--benchmark-control')",
     "        if config.get('api_key_file'):\n            argv += ['--api-key-file', config['api_key_file']]\n"
     "        if benchmark:\n            argv.append('--benchmark-control')"),
    ("if not profile['qualified_pool'] and not profile.get('serving_smoke_qualified'):",
     "if not profile['qualified_pool'] and not profile.get('serving_smoke_qualified') and not profile.get('operator_trial'):"),
    ("choices=['TF351', 'TF351-3.3M', 'TF29', '1.2.0']", "choices=sorted(p.stem for p in (HERE / 'profiles').glob('*.json'))"),
    ("    parser.add_argument('--run', action='store_true')",
     "    parser.add_argument('--run', action='store_true')\n"
     "    parser.add_argument('--check', action='store_true', help=\"validate this rank's files as --run does, start nothing\")"),
    ("benchmark=args.benchmark, validate_files=args.run)", "benchmark=args.benchmark, validate_files=args.run or args.check)"),
])
print("patched", out)
