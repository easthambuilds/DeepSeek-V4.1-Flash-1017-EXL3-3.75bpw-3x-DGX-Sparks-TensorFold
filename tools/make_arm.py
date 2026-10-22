"""make_arm.py STEP [args] : this checkpoint as an arm of LakyFx's TP3 runtime. CPU only; never starts the engine.
Standard library plus the runtime's own preparation tools (--tools, the patched copy from tools/patch_runtime.py).
Every output path must be new, except the profile, which start.sh rewrites with the serving settings.

  manifest REPO REV CKPT OUT_MANIFEST   the Hub's own list of the checkpoint's shards (size, LFS sha256) at REV, in the
                                        form the runtime's source_verify.py reads.
  kit      --tools T CKPT PLAN KIT ARM  KIT/models/ARM/config.json: the checkpoint's config through the runtime's
                                        tp3_layout.patch_config, pinned to this plan.
  models   --tools T CKPT PLAN RECEIPT KIT MODEL_DIR ARM
                                        MODEL_DIR: that config, the tokenizer, the chat template, and
                                        vision.safetensors (the 266 vision tensors, every byte compared with the
                                        checkpoint by the runtime's compare_payload) with its index.
  ranks    --tools T CKPT PLAN RECEIPT KEYS_DIR OUT_DIR ARM
                                        prints the three rank_cache.py --write and rank_verify.py --full-hash commands.
  profile  OUT ARM                      the launcher profile, from the environment: POOL CONTEXT PARALLEL ROUND_ROWS
                                        PROMPT_CACHE_GIB REASONING SERVED_NAME QUANT_REPO QUANT_REV ROPE_ROWS
                                        CHAT_TEMPLATE_SHA256 (round_ms: the runtime's 1.2.0 profile, read from
                                        RUNTIME_PROFILE)."""
import hashlib, json, os, shutil, sys, urllib.request
from pathlib import Path

ORIG = "1bae0e1088993b9b82de08f59de5be5a34e42170"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
args = sys.argv[2:]
if args[:1] == ["--tools"]:
    sys.path.insert(0, args[1]); TOOLS = Path(args[1]); args = args[2:]


def manifest(repo, rev, ckpt, out):
    hub = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    with urllib.request.urlopen(f"{hub}/api/models/{repo}/tree/{rev}?recursive=false", timeout=60) as r:
        tree = json.load(r)
    shards = sorted((e for e in tree if e["type"] == "file" and e["path"].endswith(".safetensors")), key=lambda e: e["path"])
    local = sorted(p.name for p in Path(ckpt).glob("*.safetensors"))
    if [e["path"] for e in shards] != local:
        sys.exit("the checkpoint's shards differ from the Hub's list")
    doc = {"id": repo, "sha": rev, "siblings": [{"rfilename": e["path"], "size": e["size"],
                                                 "lfs": {"sha256": e["lfs"]["oid"]}} for e in shards]}
    with open(out, "x") as f:
        json.dump(doc, f, indent=1)
    print(f"{len(shards)} shards listed at {rev[:8]}")


def kit(ckpt, plan, kit, arm):
    from tp3_layout import patch_config
    rope = json.loads((Path(kit) / "rope.json").read_text())
    p = json.loads(Path(plan).read_bytes())
    cfg = patch_config(json.loads((Path(ckpt) / "config.json").read_text()))
    cfg["tensorfold_tp3"].update(arm=arm, weight_plan_sha256=sha(plan), sources=p["sources"], original_revision=ORIG)
    d = Path(kit) / "models" / arm
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(json.dumps({"rope_rows": rope["rows"], "config_sha256": sha(d / "config.json")}))


def models(ckpt, plan, receipt, kit, model_dir, arm):
    from materialize_checkpoint import write_shard
    from source_verify import check_receipt
    from vision_export import selected, compare_payload
    ckpt, model_dir = Path(ckpt), Path(model_dir)
    p = json.loads(Path(plan).read_bytes()); items = selected(p); rc = json.loads(Path(receipt).read_bytes())
    shard = next(iter(items.values()))["shard"]
    check_receipt(rc, ckpt, p["sources"]["mia"], {shard})
    model_dir.mkdir(parents=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copyfile(ckpt / name, model_dir / name)
    shutil.copyfile(Path(kit) / "models" / arm / "config.json", model_dir / "config.json")
    target = model_dir / "vision.safetensors"
    header, payload = write_shard(ckpt / shard, target, items, read_mib_s=0)
    compared = compare_payload(ckpt / shard, target, items, read_mib_s=0)
    check_receipt(rc, ckpt, p["sources"]["mia"], {shard})
    with (model_dir / "model.safetensors.index.json").open("x") as f:
        json.dump({"metadata": {"total_size": payload}, "weight_map": {n: "vision.safetensors" for n in header}}, f)
    print(json.dumps({"vision_entries": len(items), "payload": payload, "compared": compared,
                      "vision_sha256": sha(target), "chat_template_sha256": sha(model_dir / "chat_template.jinja")}))


def ranks(ckpt, plan, receipt, keys, out_dir, arm):
    p20 = sha(plan)[:20]
    for r in range(3):
        out = Path(out_dir) / f"rank{r}of3-{arm}-{p20}.bin"
        common = ["--plan", plan, "--keys", f"{keys}/rank{r}.keys", "--pollard", ckpt, "--mia", ckpt]
        print(" ".join(["python3", "-E", "-s", str(TOOLS / "rank_cache.py"), *common, "--out", str(out),
                        "--source-receipt", f"mia={receipt}", "--write"]))
        print(" ".join(["python3", "-E", "-s", str(TOOLS / "rank_verify.py"), *common, "--artifact", str(out),
                        "--out", str(out.with_suffix(".verified.json")), "--full-hash", "--source-receipt", f"mia={receipt}"]))


def profile(out, arm):
    e = os.environ
    base = json.loads(Path(e["RUNTIME_PROFILE"]).read_text())
    doc = {"schema": "ds41f.tensorfold-profile.v1", "version": "eb-tp3a-recipe", "arm": arm,
           "model_revision": "DeepSeek-V4.1-Flash-1017", "source_repo": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
           "source_revision": ORIG, "hub_repo": e["QUANT_REPO"], "hub_revision": e["QUANT_REV"],
           "chat_template_sha256": e["CHAT_TEMPLATE_SHA256"], "rank_cache_subdir": f"weights/{arm}",
           "engram_subdir": "engram", "engram_marker": "SOURCE.json", "routed_expert_bpw": 3.75,
           "pool_tokens": int(e["POOL"]), "context_tokens": int(e["CONTEXT"]), "parallel": int(e["PARALLEL"]),
           "round_rows": int(e["ROUND_ROWS"]), "prompt_snapshot_gib_head": int(e["PROMPT_CACHE_GIB"]),
           "round_ms": base["round_ms"], "reasoning": e.get("REASONING") or None, "served_name": e["SERVED_NAME"],
           "qualified_pool": False, "serving_smoke_qualified": False, "operator_trial": True,
           "qualification_scope": "easthambuilds recipe: this checkpoint on LakyFx's TP3 runtime; see the README's results",
           "native_kv": base.get("native_kv"), "source_commit": base["source_commit"],
           "rope_table_reference_pool": 10000000, "rope_rows": int(e["ROPE_ROWS"]), "comparative_benchmarks_rerun": False}
    if len(doc["round_ms"].split(",")) != doc["round_rows"]:
        sys.exit("ROUND_ROWS must match the runtime profile's round_ms table (32)")
    Path(out).write_text(json.dumps(doc, indent=2) + "\n")


{"manifest": manifest, "kit": kit, "models": models, "ranks": ranks, "profile": profile}[sys.argv[1]](*args)
