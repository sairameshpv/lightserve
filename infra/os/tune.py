"""OS tuning, one setting group at a time (8i): put the machine's group in "as found" or "tuned",
proving each change took. Groups: gpu (clocks locked at max), memory (huge pages, swappiness),
network (the network card's backlog; the connection queue is not tested, see FILES); cpu_power is only reported (hidden on the VM).

  python3 infra/os/tune.py check                       what's here, and today's values
  sudo python3 infra/os/tune.py memory tuned           one group to one side, checked [--dry-run]
  sudo python3 infra/os/tune.py restore                every saved original back

The originals are saved once (--state) before the first change. Exit 1 = a change didn't take,
2 = the group can't run here: as the A/B tool's --a-setup / --b-setup, either stops the run
before a round is measured on the wrong setting.
"""
import json
import re
import subprocess
import time
from pathlib import Path

try:
    from infra.os.topology import _bracketed, _read, _run
except ImportError:  # run as a script on the VM: python3 infra/os/tune.py
    from topology import _bracketed, _read, _run

# name -> file under the machine root; huge pages' file marks the active choice in [brackets]
FILES = {
    "memory": {"thp": "sys/kernel/mm/transparent_hugepage/enabled", "swappiness": "proc/sys/vm/swappiness"},
    # the network card's incoming queue (whole machine). The connection queue (somaxconn) is NOT
    # here: each container has its own (checked on Docker's Linux), and lightserve's web server asks
    # for only 2048, below every value, so raising it changes nothing (8i review)
    "network": {"netdev_max_backlog": "proc/sys/net/core/netdev_max_backlog"},
}


TUNED = {"thp": "always", "swappiness": "10", "netdev_max_backlog": "16384"}


class CantRunHere(Exception):
    """The group's setting isn't exposed on this machine."""


class NotTaken(Exception):
    """A change was written but the machine still reads the old value."""


def _value(root, name, rel):
    text = _read(root, rel)
    # the kernel answers "always [madvise] never"; a plain word is a fake machine folder
    return (_bracketed(text) or text) if name == "thp" else text


def _write(path, text):
    path.write_text(text + "\n")


def _apply(root, group, want):
    """Write each value, then read every one back."""
    for name, value in want.items():
        _write(root / FILES[group][name], value)
    wrong = [f"{n} reads {_value(root, n, FILES[group][n])}, not {v}" for n, v in want.items()
             if _value(root, n, FILES[group][n]) != v]
    if wrong:
        raise NotTaken(f"{group}: DIDN'T TAKE: " + "; ".join(wrong))


def _smi(run, *args):
    """One nvidia-smi change; a failure (e.g. no root) is a message, not a traceback."""
    try:
        run(["nvidia-smi", *args])
    except (OSError, subprocess.CalledProcessError) as e:
        why = (getattr(e, "stderr", None) or str(e)).strip()
        raise NotTaken(f"gpu: nvidia-smi {' '.join(args)} failed: {why}")


def _gpu(side, saved, state_file, run):
    """Clocks locked at max ("tuned") or reset to the driver's own choice ("as-found"). The node
    starts with them free (idles at 210 MHz), so as found = reset (-rgc)."""
    try:
        clocks = gpu_clocks(run)
    except (OSError, ValueError, subprocess.CalledProcessError) as e:
        raise CantRunHere(f"no working nvidia-smi: {e}")
    if side == "as-found":
        if "gpu" in saved:
            _smi(run, "-rgc")
        return
    saved.setdefault("gpu", {"clocks": "free"})
    state_file.write_text(json.dumps(saved))
    top = clocks["max_sm_clock_mhz"]
    _smi(run, "-lgc", f"{top},{top}")
    now = gpu_clocks(run)["sm_clock_mhz"]
    if now < 0.9 * top:  # a little under max is normal (power or heat limits)
        raise NotTaken(f"gpu: DIDN'T TAKE: SM clock reads {now} MHz after locking at {top}")


def set_group(root, group, side, state_file, run=_run):
    """Put one group in `side` ("tuned" / "as-found"), then read every value back. The true
    originals are saved once, before anything changes (a second "tuned" keeps them); "as-found"
    writes them back and keeps them for the next round (nothing saved = already as found)."""
    root, state_file = Path(root), Path(state_file)
    saved = json.loads(state_file.read_text()) if state_file.exists() else {}
    if group == "gpu":
        return _gpu(side, saved, state_file, run)
    found = check_files(root, group)
    if side == "as-found":
        if group in saved:
            _apply(root, group, saved[group])
        return
    saved.setdefault(group, found)
    state_file.write_text(json.dumps(saved))
    _apply(root, group, {n: TUNED[n] for n in found})


def restore(root, state_file, run=_run):
    """Every saved group back to its originals, then forget them. Returns the groups restored.
    Every group is tried even if one fails (a memory write that doesn't take must not leave the
    GPU clock locked); on any failure the originals stay saved, so a second restore can retry."""
    root, state_file = Path(root), Path(state_file)
    if not state_file.exists():
        return []
    saved, failed = json.loads(state_file.read_text()), []
    for group in FILES:
        if group in saved:
            try:
                _apply(root, group, saved[group])
            except NotTaken as e:
                failed.append(str(e))
    if "gpu" in saved:
        try:
            _smi(run, "-rgc")
        except NotTaken as e:
            failed.append(str(e))
    if failed:
        raise NotTaken("restore incomplete, originals kept in " + str(state_file) + ": " + "; ".join(failed))
    state_file.unlink()
    return [g for g in (*FILES, "gpu") if g in saved]


def check_files(root, group):
    """One file-based group's current values; CantRunHere when one isn't exposed."""
    values = {n: _value(root, n, rel) for n, rel in FILES[group].items()}
    missing = [n for n, v in values.items() if v is None]
    if missing:
        raise CantRunHere(f"not exposed here: {', '.join(missing)}")
    return values


def check(root, run):
    """Each group's current values, and whether it can run here (a missing file = it can't)."""
    root, out = Path(root), {}
    for group in FILES:
        try:
            out[group] = {"available": True, "values": check_files(root, group)}
        except CantRunHere as e:
            out[group] = {"available": False, "why": str(e)}
    swap = re.search(r"^SwapTotal:\s+(\d+) kB", _read(root, "proc/meminfo") or "", re.M)
    if out["memory"]["available"] and swap:  # not a setting: only what swappiness has to act on
        kb = int(swap.group(1))
        out["memory"]["note"] = f"swap {kb // 1024} MB" if kb else "no swap: swappiness has nothing to act on"
    try:
        out["gpu"] = {"available": True, "values": gpu_clocks(run)}
    except (OSError, ValueError, subprocess.CalledProcessError) as e:
        out["gpu"] = {"available": False, "why": f"no working nvidia-smi: {e}"}
    if out["gpu"]["available"]:  # a separate query: a field this driver lacks can't block the group
        try:
            out["gpu"]["status"] = gpu_status(run)
        except (OSError, ValueError, subprocess.CalledProcessError) as e:
            out["gpu"]["status"] = {"unavailable": str(e)}
    governor = _read(root, "sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    out["cpu_power"] = {"available": False, "why": "only reported, not tuned (8i plan)" if governor
                        else "CPU speed policy not exposed (typical VM)"}
    return out


# Also logged every 5 s during VM-3 (README runbook): what the GPU did under load, not just idle.
STATUS = ("power.draw,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown,"
          "clocks_event_reasons.sw_thermal_slowdown")


def gpu_status(run):
    """GPU 0's power use (W) and whether a power, hardware or heat limit is slowing it right now."""
    f = [x.strip() for x in run(["nvidia-smi", f"--query-gpu={STATUS}",
                                 "--format=csv,noheader,nounits"]).splitlines()[0].split(",")]
    return {"power_w": float(f[0]), "power_cap": f[1], "hw_slowdown": f[2], "heat_slowdown": f[3]}


def gpu_clocks(run):
    """GPU 0's SM clock now and its max, in MHz."""
    now, top = run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm",
                    "--format=csv,noheader,nounits"]).splitlines()[0].split(",")
    return {"sm_clock_mhz": int(now), "max_sm_clock_mhz": int(top)}


def describe(root, group, side, state_file, run):
    """What `group side` would change, without changing it (--dry-run)."""
    root, state_file = Path(root), Path(state_file)
    saved = json.loads(state_file.read_text()) if state_file.exists() else {}
    if group == "gpu":
        top = check(root, run)["gpu"]
        if not top["available"]:
            raise CantRunHere(top["why"])
        if side == "tuned":
            m = top["values"]["max_sm_clock_mhz"]
            return [f"gpu: would run nvidia-smi -lgc {m},{m} (SM clock now {top['values']['sm_clock_mhz']} MHz)"]
        return ["gpu: would run nvidia-smi -rgc" if "gpu" in saved else "gpu: nothing saved, already as found"]
    found = check_files(root, group)
    if side == "as-found" and group not in saved:
        return [f"{group}: nothing saved, already as found"]
    want = saved[group] if side == "as-found" else {n: TUNED[n] for n in found}
    return [f"{group}: would set " + ", ".join(f"{n} {found[n]} -> {want[n]}" for n in found)]


def main(argv=None, run=_run):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("check", "restore", "gpu", "memory", "network"))
    ap.add_argument("side", nargs="?", choices=("as-found", "tuned"), help="for a group")
    ap.add_argument("--root", default="/", help="where /proc and /sys live (tests)")
    ap.add_argument("--state", default="/var/tmp/lightserve_tune_state.json", help="saved originals")
    ap.add_argument("--dry-run", action="store_true", help="print the change, make none")
    a = ap.parse_args(argv)
    if a.what == "check":
        for group, r in check(a.root, run).items():
            vals = ", ".join(f"{k} {v}" for k, v in {**r.get("values", {}), **r.get("status", {})}.items())
            vals += f"; {r['note']}" if "note" in r else ""
            print(f"{group}: available ({vals})" if r["available"] else f"{group}: can't run here: {r['why']}")
        state = Path(a.state)
        if state.exists():  # e.g. an aborted session: /var/tmp survives a node stop/start
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(state.stat().st_mtime))
            print(f"WARNING: originals saved {when} in {state} ({', '.join(json.loads(state.read_text()))}) "
                  "and never restored: a session didn't finish; run restore before tuning again")
        return
    if a.what == "restore":
        try:
            print("restored: " + (", ".join(restore(a.root, a.state, run)) or "nothing saved"))
        except NotTaken as e:
            print(str(e))
            raise SystemExit(1)
        return
    if a.side is None:
        ap.error(f"{a.what} needs a side: as-found or tuned")
    try:
        if a.dry_run:
            print("\n".join(describe(a.root, a.what, a.side, a.state, run)))
        else:
            set_group(a.root, a.what, a.side, a.state, run)
            print(f"{a.what}: {a.side}, checked")
    except CantRunHere as e:
        print(f"{a.what}: can't run here: {e}")
        raise SystemExit(2)
    except NotTaken as e:
        print(str(e))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
