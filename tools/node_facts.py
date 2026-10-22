"""node_facts.py: this Spark's facts for tools/topology.py, as JSON on stdout. Read-only, standard library only.

RoCE devices with their netdev, port state, IPv4 addresses and RoCE v2 GID table; page size, memlock limit,
MemAvailable, the GPU, the home directory. Runs on every Spark (the head pipes it over ssh)."""
import json, os, platform, resource, subprocess
from pathlib import Path


def run(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def ipv4(netdev):
    try:
        doc = json.loads(run("ip", "-j", "-4", "addr", "show", "dev", netdev) or "[]")
    except json.JSONDecodeError:
        return []
    return [f"{a['local']}/{a['prefixlen']}" for d in doc for a in d.get("addr_info", []) if a.get("family") == "inet"]


rdma = []
for dev in sorted(Path("/sys/class/infiniband").glob("*")):
    nets = sorted(p.name for p in (dev / "device/net").glob("*")) if (dev / "device/net").exists() else []
    port = dev / "ports/1"
    gids = {}
    for g in sorted((port / "gids").glob("*"), key=lambda p: int(p.name)) if (port / "gids").exists() else []:
        try:
            value = g.read_text().strip()
            kind = (port / "gid_attrs/types" / g.name).read_text().strip()
        except OSError:
            continue
        if value.strip("0:") and kind:
            gids[g.name] = {"gid": value, "type": kind}
    state = (port / "state").read_text().strip() if (port / "state").exists() else ""
    rdma.append({"dev": dev.name, "netdev": nets[0] if nets else None, "state": state,
                 "ipv4": ipv4(nets[0]) if nets else [], "gids": gids})

mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if l.split()[0] in ("MemTotal:", "MemAvailable:")}
soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
print(json.dumps({
    "host": platform.node(), "home": str(Path.home()), "machine": platform.machine(), "kernel": platform.release(),
    "page_size": os.sysconf("SC_PAGE_SIZE"), "memlock_unlimited": soft == resource.RLIM_INFINITY,
    "mem_total_kib": mem.get("MemTotal"), "mem_avail_kib": mem.get("MemAvailable"),
    "gpu": run("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"),
    "default_src": (run("ip", "-4", "route", "get", "1.1.1.1").split(" src ") + [""])[1].split(" ")[0],
    "rdma": rdma}))
