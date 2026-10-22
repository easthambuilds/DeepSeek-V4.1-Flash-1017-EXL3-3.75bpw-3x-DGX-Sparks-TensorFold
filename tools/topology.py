"""topology.py FACTS0 FACTS1 FACTS2 OUT_DIR : the three Sparks' RoCE fabric, from tools/node_facts.py output, as the
runtime's cluster config. Standard library only. Writes OUT_DIR/topology.json (cluster config without the serving
settings, which start.sh adds) and OUT_DIR/routes.json (the host routes the ranks need); prints a summary.

What it works out, for each rank (0 = the head, 1 and 2 = WORKER1 and WORKER2):
  - its four rails: the active RoCE devices with an IPv4 address, port 0's twins first (a cabled ConnectX-7 port
    shows up as two RoCE devices, one per PCIe function), as the runtime expects them;
  - peer masks: which rails reach which peer, by shared IPv4 subnet (a direct cable is one subnet per twin);
  - the GID index of the rails' IPv4 RoCE v2 GIDs (the runtime takes one per rank);
  - the master address (the head's address on its cable to rank 1) and each rank's NCCL socket interface (its
    address on its cable to the head), plus the address range NCCL may use;
  - routes: with one subnet per cable, rank 2 has no direct path to the head's cable to rank 1, nor ranks 1 and 2 to
    each other's socket addresses. Each such address needs a /32 route over the cable the two ranks share.
Exit 1 with the reason when the fabric does not fit (fewer than four rails, a peer with no cable, mixed GID indexes)."""
import ipaddress, json, re, sys
from pathlib import Path

facts = [json.loads(Path(p).read_text()) for p in sys.argv[1:4]]
out = Path(sys.argv[4])
errors = []


def rail_key(r):
    m = re.search(r"f(\d)np", r["netdev"] or "")
    return (int(m.group(1)) if m else 9, "P" in (r["netdev"] or "")[:5], r["dev"])


ranks = []
for i, f in enumerate(facts):
    rails = []
    for r in f["rdma"]:
        if not r["netdev"] or "ACTIVE" not in r["state"] or not r["ipv4"]:
            continue
        iface = ipaddress.ip_interface(r["ipv4"][0])
        gid = [k for k, g in r["gids"].items() if g["type"] == "RoCE v2"
               and g["gid"].lower().replace(":", "").endswith("ffff" + "".join(f"{b:02x}" for b in iface.ip.packed))]
        rails.append(dict(r, iface=iface, gid_index=int(gid[0]) if gid else None))
    rails.sort(key=rail_key)
    if len(rails) != 4:
        errors.append(f"rank {i} ({f['host']}): {len(rails)} active RoCE devices with IPv4, want 4 (two cabled ports): "
                      + ", ".join(r["dev"] for r in rails))
    ranks.append({"host": f["host"], "rails": rails})
if errors:
    sys.exit("\n".join(errors))

# which rails of rank i share a subnet with rank j
def shared(i, j):
    nets_j = {r["iface"].network for r in ranks[j]["rails"]}
    return [k for k, r in enumerate(ranks[i]["rails"]) if r["iface"].network in nets_j]

masks = [[sum(1 << k for k in shared(i, j)) if i != j else 0 for j in range(3)] for i in range(3)]
for i in range(3):
    for j in range(3):
        if i != j and not masks[i][j]:
            errors.append(f"rank {i} ({ranks[i]['host']}) shares no subnet with rank {j} ({ranks[j]['host']}): cable them")
    if masks[i][(i + 1) % 3] & masks[i][(i + 2) % 3]:
        errors.append(f"rank {i}: a rail reaches both peers (a switch?). This recipe is qualified on direct cables only")
gids = [{r["gid_index"] for r in rk["rails"]} for rk in ranks]
for i, g in enumerate(gids):
    if None in g or len(g) != 1:
        errors.append(f"rank {i} ({ranks[i]['host']}): the rails' IPv4 RoCE v2 GID indexes are {sorted(map(str, g))}; "
                      "the runtime needs one index for all four (check show_gids)")
if errors:
    sys.exit("\n".join(errors))

def addr(i, k):
    return str(ranks[i]["rails"][k]["iface"].ip)

# the head's address on its first rail to rank 1 is the master; each rank's NCCL socket is its first rail to the head
r01 = shared(0, 1)[0]
master = addr(0, r01)
sock = {0: r01, 1: shared(1, 0)[0], 2: shared(2, 0)[0]}
nets = [r["iface"].network for rk in ranks for r in rk["rails"]]
rng = nets[0]
while not all(n.subnet_of(rng) for n in nets):
    rng = rng.supernet()
lan = [f["default_src"] for f in facts if f["default_src"] and ipaddress.ip_address(f["default_src"]) in rng]
if lan:
    sys.exit(f"the fabric address range {rng} also covers a LAN address ({', '.join(lan)}): give the cables their own subnets")

# routes: for each rank, every peer address it must reach (the master; the other ranks' socket addresses)
routes = []
for i in range(3):
    connected = {r["iface"].network for r in ranks[i]["rails"]}
    want = {master: 0} if i else {}
    for j in range(3):
        if j != i:
            want[addr(j, sock[j])] = j
    for a, j in want.items():
        if any(ipaddress.ip_address(a) in n for n in connected):
            routes.append({"rank": i, "to": a, "dev": None, "via": None})
            continue
        k = shared(i, j)[0]
        net = ranks[i]["rails"][k]["iface"].network
        via = next(str(r["iface"].ip) for r in ranks[j]["rails"] if r["iface"].network == net)
        routes.append({"rank": i, "to": a, "dev": ranks[i]["rails"][k]["netdev"], "via": via})

topo = {"schema": "ds41f.tensorfold-topology.v1", "master_address": master, "nccl_interface": ranks[0]["rails"][r01]["netdev"],
        "nccl_hcas": [r["dev"] for r in ranks[0]["rails"]], "nccl_address_range": str(rng),
        "ranks": [{"rank": i, "host": ranks[i]["host"], "rdma_devices": [r["dev"] for r in ranks[i]["rails"]],
                   "gid_index": gids[i].pop(), "peer_masks": masks[i], "nccl_interface": ranks[i]["rails"][sock[i]]["netdev"],
                   "nccl_hcas": [r["dev"] for r in ranks[i]["rails"]],
                   "addresses": [str(r["iface"]) for r in ranks[i]["rails"]]} for i in range(3)]}
out.mkdir(parents=True, exist_ok=True)
(out / "topology.json").write_text(json.dumps(topo, indent=2) + "\n")
(out / "routes.json").write_text(json.dumps(routes, indent=2) + "\n")
for rk in topo["ranks"]:
    print(f"   rank {rk['rank']} {rk['host']}: rails {','.join(rk['rdma_devices'])} GID {rk['gid_index']}"
          f" masks {rk['peer_masks']} socket {rk['nccl_interface']}")
print(f"   master {master}, NCCL range {rng}; {sum(1 for r in routes if r['dev'])} host routes needed")
