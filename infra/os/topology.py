"""Look before tuning: a map of the GPU machine (CPU groups, twin cores, the GPU's nearby cores,
interrupts, current GPU/OS settings) and what it means for pinning lightserve (8h) and the OS
tuning A/Bs (8i). Reads Linux's own files under --root (default /), so tests run it on fake
machine folders. Python 3 standard library only: nothing to install on the GPU VM.

    python3 infra/os/topology.py --out runs/topology
"""
import argparse
import json
import re
import subprocess
from pathlib import Path


def parse_cpulist(text):
    """Linux's core-list format: "0-3,8-11" -> [0, 1, 2, 3, 8, 9, 10, 11]."""
    cpus = []
    for part in text.strip().split(","):
        if part:
            lo, _, hi = part.partition("-")
            cpus += range(int(lo), int(hi or lo) + 1)
    return cpus


def _read(root, rel):
    """A file's stripped text, or None when the machine doesn't expose it."""
    try:
        return (root / rel).read_text().strip()
    except OSError:
        return None


def _int(text):
    return None if text is None else int(text)


def _bracketed(text):
    """THP files mark the active choice: "always [madvise] never" -> "madvise"."""
    m = re.search(r"\[(\S+)\]", text or "")
    return m.group(1) if m else None


def _isolated(cmdline):
    """isolcpus=[flags,]<cpu-list>: the flags are words, the rest is the core list."""
    m = re.search(r"(?:^|\s)isolcpus=(\S+)", cmdline)
    if not m:
        return []
    return parse_cpulist(",".join(p for p in m.group(1).split(",") if p[:1].isdigit()))


def _irqs(root):
    """Steerable (numbered) interrupts: device name, total count, the cores allowed to handle it."""
    lines = (_read(root, "proc/interrupts") or "").splitlines()
    if not lines:
        return []
    n_cpus = len(lines[0].split())
    out = []
    for line in lines[1:]:
        head, _, rest = line.partition(":")
        if not head.strip().isdigit():
            continue
        parts = rest.split()
        irq = int(head)
        out.append({"irq": irq, "device": parts[-1], "count": sum(int(x) for x in parts[:n_cpus]),
                    "cpus": parse_cpulist(_read(root, f"proc/irq/{irq}/smp_affinity_list") or "")})
    return out


def gather(root=Path("/")):
    """Everything the report needs, from files under `root`."""
    root = Path(root)
    cpu_dir = "sys/devices/system/cpu"
    cpus = parse_cpulist(_read(root, f"{cpu_dir}/online") or "")
    siblings = {c: parse_cpulist(_read(root, f"{cpu_dir}/cpu{c}/topology/thread_siblings_list") or str(c)) for c in cpus}
    numa = {}
    for d in sorted((root / "sys/devices/system/node").glob("node[0-9]*")):
        numa[int(d.name[4:])] = parse_cpulist(_read(root, f"sys/devices/system/node/{d.name}/cpulist") or "")
    idle = sorted((root / f"{cpu_dir}/cpu0/cpuidle").glob("state[0-9]*"), key=lambda p: int(p.name[5:]))
    cpuinfo = _read(root, "proc/cpuinfo") or ""
    model = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.M)
    flags = re.search(r"^flags\s*:\s*(.+)$", cpuinfo, re.M)
    cmdline = _read(root, "proc/cmdline") or ""
    comms = {_read(root, str(p.relative_to(root))) for p in root.glob("proc/[0-9]*/comm")}
    return {
        "cpu_model": model.group(1) if model else None,
        "virtual": ("hypervisor" in flags.group(1).split()) if flags else None,  # x86's flag; None = unknown
        "cpus": cpus,
        "cores": [list(t) for t in sorted({tuple(s) for s in siblings.values()})],  # twin groups
        "numa": numa,
        "governor": _read(root, f"{cpu_dir}/cpu0/cpufreq/scaling_governor"),
        "idle_states": [_read(root, str(p.relative_to(root) / "name")) for p in idle] or None,
        "thp": {k: _bracketed(_read(root, f"sys/kernel/mm/transparent_hugepage/{k}")) for k in ("enabled", "defrag")},
        "swappiness": _int(_read(root, "proc/sys/vm/swappiness")),
        "somaxconn": _int(_read(root, "proc/sys/net/core/somaxconn")),
        "netdev_max_backlog": _int(_read(root, "proc/sys/net/core/netdev_max_backlog")),
        "cmdline": cmdline,
        "isolated": _isolated(cmdline),
        "irqs": _irqs(root),
        "irqbalance": "irqbalance" in comms,
    }


# --- the GPU part: nvidia-smi, plus where the GPU sits from /sys ---

QUERY = ("index,name,pci.bus_id,persistence_mode,clocks.sm,clocks.max.sm,clocks.applications.graphics,"
         "power.limit,power.max_limit,driver_version")


def _num(text, kind):
    return None if text.startswith("[") else kind(text)  # "[N/A]", "[Not Supported]" -> None


def parse_gpus(csv_text):
    """nvidia-smi --query-gpu=QUERY --format=csv,noheader,nounits -> one dict per GPU."""
    gpus = []
    for line in csv_text.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(f[0]), "name": f[1], "pci": f[2], "persistence": f[3] == "Enabled",
                     "sm_clock_mhz": _num(f[4], int), "max_sm_clock_mhz": _num(f[5], int),
                     "app_clock_mhz": _num(f[6], int), "power_limit_w": _num(f[7], float),
                     "max_power_limit_w": _num(f[8], float), "driver": f[9]})
    return gpus


def sysfs_pci_address(bus_id):
    """nvidia-smi's "00000000:8D:00.0" -> /sys's "0000:8d:00.0" (4-digit domain, lower case)."""
    domain, rest = bus_id.split(":", 1)
    return f"{domain[-4:]}:{rest}".lower()


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def gather_gpus(root=Path("/"), run=_run):
    """Each GPU's settings and its NUMA node + nearby cores (None = unknown: VMs often say -1);
    `nvidia-smi topo -m` kept as raw text. No nvidia-smi or an error: available False + why."""
    root = Path(root)
    try:
        gpus = parse_gpus(run(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"]))
        topo = run(["nvidia-smi", "topo", "-m"])
    except FileNotFoundError:
        return {"available": False, "gpus": [], "topo": None, "error": "nvidia-smi not found"}
    except subprocess.CalledProcessError as e:  # e.g. a query field this driver doesn't know: show why
        return {"available": False, "gpus": [], "topo": None, "error": (e.stderr or e.stdout or str(e)).strip()}
    for g in gpus:
        dev = f"sys/bus/pci/devices/{sysfs_pci_address(g['pci'])}"
        node = _int(_read(root, f"{dev}/numa_node"))
        g["numa_node"] = None if node is None or node < 0 else node
        g["local_cpus"] = parse_cpulist(_read(root, f"{dev}/local_cpulist") or "")
    return {"available": True, "gpus": gpus, "topo": topo, "error": None}


# --- what it means ---

def _numa_verdict(facts, gpu):
    n = len(facts["numa"])
    if n == 0:
        return "NUMA layout unknown (no /sys/devices/system/node)"
    if n == 1:
        return ("1 NUMA node: every core is equally close to memory and the GPU, so NUMA pinning can't help "
                "here; pinning is still worth it to keep other work off the engine's core (8h)")
    if gpu and gpu["numa_node"] is not None:
        k = gpu["numa_node"]
        return f"{n} NUMA nodes; the GPU is on node {k}: keep lightserve there (numactl --cpunodebind={k} --membind={k})"
    return f"{n} NUMA nodes, but the GPU's node is unknown (the VM says -1): use nvidia-smi topo -m's CPU Affinity"


def _layout(facts, gpu):
    """Whole physical cores (twins together): housekeeping = cpu0's core (interrupts, the OS);
    engine = the first core near the GPU, its twin left idle (lightserve's GPU-feeding thread);
    server = the next core (2 threads), or 2 cores without twins (the web server); rest = the rest."""
    groups = sorted(facts["cores"])
    near = set(gpu["local_cpus"]) if gpu and gpu["numa_node"] is not None and gpu["local_cpus"] else set(facts["cpus"])
    hk = next((g for g in groups if 0 in g), groups[0] if groups else None)
    free = [g for g in groups if g != hk and set(g) <= near]
    n_server = 1 if hk and len(hk) > 1 else 2
    if hk is None or len(free) < 1 + n_server:
        return None, f"too few cores near the GPU ({len(free)} free physical cores, need {1 + n_server})"
    engine, server = free[0], [c for g in free[1:1 + n_server] for c in g]
    used = set(hk) | set(engine) | set(server)
    return {"housekeeping": hk, "engine": engine[0], "engine_twin_idle": engine[1:], "server": sorted(server),
            "rest": [c for c in facts["cpus"] if c not in used]}, None


def _tuning(facts, gpus):
    """The current value of each 8i setting group, and what its A/B would change."""
    g = gpus["gpus"][0] if gpus["gpus"] else None
    gpu = ("no GPU tools (nvidia-smi) here" if g is None else
           f"persistence {'on' if g['persistence'] else 'off'}, SM clock now {g['sm_clock_mhz']} MHz (max "
           f"{g['max_sm_clock_mhz']}); A/B: persistence on + clocks locked at max (nvidia-smi -pm 1, "
           f"-lgc {g['max_sm_clock_mhz']},{g['max_sm_clock_mhz']})")
    if facts["governor"] is None and facts["idle_states"] is None:
        cpu = "CPU speed policy (governor) and sleep states not exposed (typical VM): this A/B can't run here"
    else:
        cpu = (f"governor {facts['governor'] or 'not exposed'}, sleep states {facts['idle_states'] or 'not exposed'}; "
               "A/B: performance governor, deep sleep states off")
    thp = facts["thp"]
    memory = (f"huge pages (THP) enabled={thp['enabled']} defrag={thp['defrag']}, swappiness {facts['swappiness']}; "
              "A/B: THP always vs now, swappiness 10")
    network = (f"somaxconn {facts['somaxconn']}, netdev_max_backlog {facts['netdev_max_backlog']}; "
               "A/B: raise both + the open-file limit")
    return {"gpu": gpu, "cpu_power": cpu, "memory": memory, "network": network}


def advise(facts, gpus):
    """What the map means for 8h (pinning) and 8i (tuning A/Bs)."""
    gpu = gpus["gpus"][0] if gpus["gpus"] else None
    layout, why = _layout(facts, gpu)
    reserved = set() if layout is None else {layout["engine"], *layout["engine_twin_idle"], *layout["server"]}
    return {
        "numa_nodes": len(facts["numa"]),
        "numa": _numa_verdict(facts, gpu),
        "layout": layout,
        "layout_why": why,
        "irqs_to_move": [{"irq": i["irq"], "device": i["device"]} for i in facts["irqs"] if reserved & set(i["cpus"])],
        "irqbalance": ("irqbalance is running: it would move interrupts back onto the reserved cores; stop it "
                       "for 8h (or ban those cores: IRQBALANCE_BANNED_CPULIST)") if facts["irqbalance"] else None,
        "tuning": _tuning(facts, gpus),
    }



# --- the report ---

def render(facts, gpus, advice):
    """The plain-text report: the map, NVIDIA's own map, then what it means."""
    f, a = facts, advice
    smt = max((len(c) for c in f["cores"]), default=1)
    lines = ["THE MACHINE",
             f"  {f['cpu_model'] or 'CPU model unknown'}, {len(f['cpus'])} CPUs = {len(f['cores'])} physical cores "
             f"x {smt} threads; virtual machine: {({True: 'yes', False: 'no', None: 'unknown'})[f['virtual']]}",
             "  NUMA nodes: " + ("; ".join(f"node {n}: {_cpulist(c)}" for n, c in f["numa"].items()) or "unknown"),
             "  twins (threads sharing a physical core): " + " ".join("+".join(map(str, c)) for c in f["cores"]),
             f"  already set aside at boot (isolcpus): {_cpulist(f['isolated']) or 'none'}",
             f"  kernel command line: {f['cmdline']}",
             "", "GPU"]
    if gpus["available"]:
        for g in gpus["gpus"]:
            lines.append(f"  GPU {g['index']}: {g['name']} at {g['pci']}, driver {g['driver']}, NUMA node "
                         f"{'unknown' if g['numa_node'] is None else g['numa_node']}, nearby cores "
                         f"{_cpulist(g['local_cpus']) or 'unknown'}")
        lines += ["", "  nvidia-smi topo -m:"] + ["    " + x for x in gpus["topo"].rstrip().splitlines()]
    else:
        lines.append(f"  no GPU information: {gpus['error']}")
    lines += ["", "INTERRUPTS (steerable)"]
    lines += [f"  {i['irq']:>4} {i['device']:<28} {i['count']:>12,} so far, may run on {_cpulist(i['cpus'])}"
              for i in sorted(f["irqs"], key=lambda i: -i["count"])[:15]]
    lines += ["", "WHAT IT MEANS", f"  NUMA: {a['numa']}"]
    lay = a["layout"]
    if lay:
        lines += ["  suggested layout for 8h (whole physical cores):",
                  f"    housekeeping (interrupts, the OS): {_cpulist(lay['housekeeping'])}",
                  f"    engine: {lay['engine']} (its twin {_cpulist(lay['engine_twin_idle']) or '-'} left idle)",
                  f"    server: {_cpulist(lay['server'])}",
                  f"    rest (everything else): {_cpulist(lay['rest'])}"]
    else:
        lines.append(f"  no layout: {a['layout_why']}")
    if a["irqs_to_move"]:
        lines.append("  interrupts to move to housekeeping: " + ", ".join(f"{i['irq']} ({i['device']})" for i in a["irqs_to_move"]))
    if a["irqbalance"]:
        lines.append(f"  {a['irqbalance']}")
    lines.append("  tuning A/Bs (8i), current values:")
    lines += [f"    {k}: {v}" for k, v in a["tuning"].items()]
    return "\n".join(lines)


def _cpulist(cpus):
    """[0, 1, 2, 5] -> "0-2,5"."""
    out, cpus = [], sorted(cpus)
    i = 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        out.append(str(cpus[i]) if i == j else f"{cpus[i]}-{cpus[j]}")
        i = j + 1
    return ",".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/", help="read /sys and /proc under this folder (tests: a fake machine)")
    ap.add_argument("--out", help="also write report.txt and topology.json (for 8h's scripts) to this folder")
    args = ap.parse_args()
    facts, gpus = gather(args.root), gather_gpus(args.root)
    advice = advise(facts, gpus)
    text = render(facts, gpus, advice)
    print(text)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.txt").write_text(text + "\n")
        (out / "topology.json").write_text(json.dumps({"facts": facts, "gpus": gpus, "advice": advice}, indent=1))


if __name__ == "__main__":
    main()
