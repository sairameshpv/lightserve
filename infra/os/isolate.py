"""CPU isolation for the noisy-neighbour experiment (8h): put the machine in one of four states,
from the topology report's layout (infra/os/topology.py --out ... -> topology.json):

  quiet           lightserve alone, unpinned
  noisy           + a neighbour burning every core but housekeeping (the OS and the load
                  generator keep those in every state: run ab_run.py under taskset on them)
  isolated-quiet  lightserve pinned (engine thread alone on its core, its twin idle; the web
                  server on the server cores); interrupts moved to housekeeping; irqbalance off
  isolated-noisy  the same + the neighbour, kept to the "rest" cores
  restore         everything stopped; interrupts and irqbalance put back

Each state is a fresh start (both containers recreated), so it's the A/B tool's --a-setup /
--b-setup:  sudo python3 infra/os/isolate.py isolated-noisy --topology runs/topology/topology.json
Interrupt changes need root; the originals are saved to a file and restored by every
non-isolated state, so alternating A/B rounds don't leak into each other.
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

try:
    from infra.os.topology import _cpulist, _irqs, gather, parse_cpulist
except ImportError:  # run as a script on the VM: python3 infra/os/isolate.py
    from topology import _cpulist, _irqs, gather, parse_cpulist

MODES = ("quiet", "noisy", "isolated-quiet", "isolated-noisy", "restore")
STOP = "docker rm -f lightserve noise 2>/dev/null || true"  # older Docker errors on a missing one


@dataclass
class Config:
    image: str = "ghcr.io/sairameshpv/lightserve:gpu-latest"
    gpus: bool = True
    server_args: str = "--n-layers 4 --max-model-len 512"
    port: int = 8000
    noise_image: str = "python:3.11-slim"


def _noise(cpus, cfg):
    """One spinning Python process per allowed core: a plain CPU burn (like stress-ng --cpu)."""
    return (f"docker run -d --name noise --cpuset-cpus {_cpulist(cpus)} {cfg.noise_image} "
            f"sh -c 'for i in $(seq {len(cpus)}); do python -c \"while 1: pass\" & done; wait'")


def _lightserve(cfg, layout=None):
    gpus = " --gpus all" if cfg.gpus else ""
    if layout is None:
        return f"docker run -d --name lightserve{gpus} -p {cfg.port}:8000 {cfg.image} {cfg.server_args}"
    cores = [layout["engine"], *layout["server"]]  # the engine's twin is in neither set: it stays idle
    return (f"docker run -d --name lightserve{gpus} -p {cfg.port}:8000 --cpuset-cpus {_cpulist(cores)} "
            f"{cfg.image} {cfg.server_args} --engine-cpus {layout['engine']} --server-cpus {_cpulist(layout['server'])}")


def plan(mode, layout, all_cpus, cfg):
    """The commands for `mode`, and what to do with interrupts ("steer" / "restore")."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; one of {', '.join(MODES)}")
    isolated = mode.startswith("isolated")
    cmds = [STOP]
    if mode == "noisy":  # never on housekeeping: the OS and the load generator live there in every state
        cmds.append(_noise([c for c in all_cpus if c not in layout["housekeeping"]], cfg))
    if mode == "isolated-noisy":
        cmds.append(_noise(layout["rest"], cfg))
    if mode != "restore":
        cmds.append(_lightserve(cfg, layout if isolated else None))
    return {"commands": cmds, "irqs": "steer" if isolated else "restore"}


def steer_irqs(root, layout, state_file, irqbalance_running):
    """Move every interrupt that may run on lightserve's cores onto the housekeeping cores. The
    true originals are saved once (a second steer keeps them). Interrupts Linux refuses to move
    (kernel-managed ones) are reported, not fatal."""
    root, state_file = Path(root), Path(state_file)
    reserved = {layout["engine"], *layout["engine_twin_idle"], *layout["server"]}
    target = _cpulist(layout["housekeeping"])
    saved = json.loads(state_file.read_text()) if state_file.exists() else {"irqs": {}, "irqbalance_was_running": irqbalance_running}
    moved, failed = [], []
    for irq in _irqs(root):
        if not reserved & set(irq["cpus"]):
            continue
        path = root / f"proc/irq/{irq['irq']}/smp_affinity_list"
        try:
            original = path.read_text().strip()
            path.write_text(target + "\n")
        except OSError:
            failed.append(irq["irq"])
            continue
        saved["irqs"].setdefault(str(irq["irq"]), original)
        moved.append(irq["irq"])
    state_file.write_text(json.dumps(saved))
    return {"moved": moved, "failed": failed}


def restore_irqs(root, state_file):
    """Put the saved interrupt cores back and forget them; say whether irqbalance should restart."""
    root, state_file = Path(root), Path(state_file)
    if not state_file.exists():
        return {"restored": [], "start_irqbalance": False}
    saved = json.loads(state_file.read_text())
    restored = []
    for irq, cpus in saved["irqs"].items():
        try:
            (root / f"proc/irq/{irq}/smp_affinity_list").write_text(cpus + "\n")
            restored.append(int(irq))
        except OSError:
            pass  # the device went away: nothing to put back
    state_file.unlink()
    return {"restored": sorted(restored), "start_irqbalance": saved["irqbalance_was_running"]}



# --- did it take effect? ---

THREADS = "docker exec lightserve sh -c 'for t in /proc/1/task/*; do echo \"== $t\"; cat $t/status; done'"


def parse_task_status(dump):
    """THREADS' output -> {thread id: its allowed cores} (inside the container the server is pid 1)."""
    threads, tid = {}, None
    for line in dump.splitlines():
        if line.startswith("== "):
            tid = int(line.rsplit("/", 1)[1])
        elif line.startswith("Cpus_allowed_list:") and tid is not None:
            threads[tid] = parse_cpulist(line.split(":", 1)[1])
    return threads


def check_pinning(threads, layout):
    """Exactly one thread alone on the engine core; every other thread on the server cores."""
    engine, server = [layout["engine"]], sorted(layout["server"])
    on_engine = [t for t, c in threads.items() if c == engine]
    stray = {t: c for t, c in threads.items() if c not in (engine, server)}
    if len(on_engine) != 1:
        return {"ok": False, "threads": threads,
                "why": f"{len(on_engine)} threads alone on the engine core {engine[0]} (want exactly 1)"}
    if stray:
        return {"ok": False, "threads": threads, "why": f"threads on unexpected cores: {stray}"}
    return {"ok": True, "threads": threads,
            "why": f"engine thread {on_engine[0]} alone on core {engine[0]}; {len(threads) - 1} others on {_cpulist(server)}"}


def _sh(cmd):
    print("+ " + cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)


def _wait_healthy(port, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5)
            return
        except OSError:
            time.sleep(1)
    raise SystemExit(f"lightserve not healthy after {timeout_s} s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=MODES)
    ap.add_argument("--topology", required=True, help="topology.json from infra/os/topology.py --out")
    ap.add_argument("--state", default="/var/tmp/lightserve_isolate_state.json", help="saved interrupt originals")
    ap.add_argument("--root", default="/", help="where /proc lives (tests)")
    ap.add_argument("--image", default=Config.image)
    ap.add_argument("--no-gpus", action="store_true", help="no --gpus all (the Mac rehearsal)")
    ap.add_argument("--server-args", default=Config.server_args)
    ap.add_argument("--port", type=int, default=Config.port)
    ap.add_argument("--no-irqs", action="store_true", help="print the interrupt changes instead (no root / the Mac)")
    ap.add_argument("--health-timeout-s", type=float, default=300)
    ap.add_argument("--dry-run", action="store_true", help="print everything, change nothing")
    args = ap.parse_args(argv)

    topo = json.loads(Path(args.topology).read_text())
    layout, all_cpus = topo["advice"]["layout"], topo["facts"]["cpus"]
    cfg = Config(image=args.image, gpus=not args.no_gpus, server_args=args.server_args, port=args.port)
    p = plan(args.mode, layout, all_cpus, cfg)
    irq_line = (f"interrupts: steer to {_cpulist(layout['housekeeping'])} (irqbalance stopped first if running)"
                if p["irqs"] == "steer" else f"interrupts: put back from {args.state} (if saved)")
    if args.dry_run:
        print("\n".join(p["commands"] + [irq_line]))
        return

    _sh(p["commands"][0])                       # stop both containers first
    if args.no_irqs:
        print("(not done) " + irq_line)
    elif p["irqs"] == "steer":
        running = gather(args.root)["irqbalance"]
        if running:
            _sh("systemctl stop irqbalance")    # it would move them straight back
        print(f"interrupts: {steer_irqs(args.root, layout, args.state, running)}")
    else:
        r = restore_irqs(args.root, args.state)
        if r["start_irqbalance"]:
            _sh("systemctl start irqbalance")
        print(f"interrupts: {r}")
    for cmd in p["commands"][1:]:
        _sh(cmd)
    if args.mode == "restore":
        return
    _wait_healthy(args.port, args.health_timeout_s)
    if args.mode.startswith("isolated"):
        r = check_pinning(parse_task_status(subprocess.run(THREADS, shell=True, check=True, capture_output=True,
                                                           text=True).stdout), layout)
        print(("pinning OK: " if r["ok"] else "PINNING WRONG: ") + r["why"])
        if not r["ok"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
