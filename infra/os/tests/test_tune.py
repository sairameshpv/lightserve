"""Tests for infra/os/tune.py: one setting group at a time, "as found" vs "tuned", on a fake
machine folder (test_topology.make_machine: huge pages madvise, swappiness 60, backlog 1000, like
the real node) and a fake nvidia-smi."""
import json
import subprocess

import pytest

import infra.os.tune as tune
from infra.os.tune import CantRunHere, NotTaken, check, main, restore, set_group
from test_topology import make_machine


def no_gpu(cmd):
    raise FileNotFoundError("nvidia-smi")


class FakeSmi:
    """nvidia-smi on an L40S that idles at 210 MHz (max 2520); `ignores` = a lock that doesn't hold."""
    def __init__(self, ignores=False):
        self.clock, self.ignores, self.changes, self.no_status, self.fails = 210, ignores, [], False, ()

    def __call__(self, cmd):
        if cmd[1].startswith("--query-gpu=power.draw"):
            if self.no_status:
                raise ValueError("Field \"clocks_event_reasons.hw_slowdown\" is not a valid field to query.")
            return "61.3, Not Active, Not Active, Not Active\n"
        if cmd[1].startswith("--query-gpu"):
            return f"{self.clock}, 2520\n"
        self.changes.append(cmd[1:])
        if cmd[1] in self.fails:
            raise subprocess.CalledProcessError(1, cmd, stderr="Insufficient Permissions")
        if cmd[1] == "-lgc" and not self.ignores:
            self.clock = int(cmd[2].split(",")[0])
        if cmd[1] == "-rgc":
            self.clock = 210
        return ""


def test_check_reads_each_group_and_says_what_can_run(tmp_path):
    make_machine(tmp_path)
    r = check(tmp_path, run=no_gpu)
    assert r["memory"] == {"available": True, "values": {"thp": "madvise", "swappiness": "60"}}
    assert r["network"] == {"available": True, "values": {"netdev_max_backlog": "1000"}}
    assert r["gpu"]["available"] is False and "nvidia-smi" in r["gpu"]["why"]
    assert r["cpu_power"]["available"] is False  # hidden by the VM, as on the real node


def test_a_missing_setting_file_means_cant_run_here(tmp_path):
    make_machine(tmp_path)
    (tmp_path / "proc/sys/net/core/netdev_max_backlog").unlink()   # as inside a container
    r = check(tmp_path, run=no_gpu)
    assert r["network"]["available"] is False and "netdev_max_backlog" in r["network"]["why"]


def test_the_connection_queue_is_not_a_machine_setting(tmp_path):
    """somaxconn belongs to each container's own network (a container reads its own default
    whatever the machine says; checked on Docker's Linux), so it's isolate.py --somaxconn's job."""
    assert "somaxconn" not in tune.TUNED and all("somaxconn" not in f for f in tune.FILES["network"].values())


def test_tuned_writes_new_values_and_saves_the_originals(tmp_path):
    make_machine(tmp_path)
    state = tmp_path / "state.json"
    set_group(tmp_path, "memory", "tuned", state)
    assert (tmp_path / "sys/kernel/mm/transparent_hugepage/enabled").read_text().strip() == "always"
    assert (tmp_path / "proc/sys/vm/swappiness").read_text().strip() == "10"
    set_group(tmp_path, "network", "tuned", state)
    assert (tmp_path / "proc/sys/net/core/somaxconn").read_text().strip() == "4096"  # untouched: see next test
    assert (tmp_path / "proc/sys/net/core/netdev_max_backlog").read_text().strip() == "16384"
    assert json.loads(state.read_text()) == {"memory": {"thp": "madvise", "swappiness": "60"},
                                             "network": {"netdev_max_backlog": "1000"}}


def test_a_second_tuned_keeps_the_true_originals(tmp_path):
    make_machine(tmp_path)
    state = tmp_path / "state.json"
    set_group(tmp_path, "memory", "tuned", state)
    set_group(tmp_path, "memory", "tuned", state)   # the machine now reads "always"/10: not originals
    assert json.loads(state.read_text())["memory"] == {"thp": "madvise", "swappiness": "60"}


def test_a_change_the_machine_ignores_stops_with_didnt_take(tmp_path, monkeypatch):
    """A VM can accept a write and keep the old value: that round must not be measured."""
    make_machine(tmp_path)
    monkeypatch.setattr(tune, "_write", lambda path, text: None)
    with pytest.raises(NotTaken, match="swappiness"):
        set_group(tmp_path, "memory", "tuned", tmp_path / "state.json")


def test_tuning_a_group_that_cant_run_here_refuses(tmp_path):
    make_machine(tmp_path)
    (tmp_path / "proc/sys/net/core/netdev_max_backlog").unlink()
    with pytest.raises(CantRunHere, match="netdev_max_backlog"):
        set_group(tmp_path, "network", "tuned", tmp_path / "state.json")


def test_as_found_puts_the_saved_originals_back(tmp_path):
    make_machine(tmp_path)
    state = tmp_path / "state.json"
    set_group(tmp_path, "memory", "tuned", state)
    set_group(tmp_path, "memory", "as-found", state)
    assert check(tmp_path, run=no_gpu)["memory"]["values"] == {"thp": "madvise", "swappiness": "60"}
    assert json.loads(state.read_text())["memory"] == {"thp": "madvise", "swappiness": "60"}  # kept for the next round


def test_as_found_with_nothing_saved_changes_nothing(tmp_path):
    """The A/B tool's first round can be an "as found" one: the machine already is."""
    make_machine(tmp_path)
    state = tmp_path / "state.json"
    set_group(tmp_path, "network", "as-found", state)
    assert check(tmp_path, run=no_gpu)["network"]["values"] == {"netdev_max_backlog": "1000"}
    assert not state.exists()


def test_alternating_rounds_then_restore_leave_the_machine_as_it_started(tmp_path):
    make_machine(tmp_path)
    state = tmp_path / "state.json"
    before = check(tmp_path, run=no_gpu)
    for side in ("tuned", "as-found", "tuned"):
        set_group(tmp_path, "memory", side, state)
        set_group(tmp_path, "network", side, state)
    assert restore(tmp_path, state) == ["memory", "network"]
    assert check(tmp_path, run=no_gpu) == before and not state.exists()


def test_restore_with_nothing_saved_is_a_no_op(tmp_path):
    make_machine(tmp_path)
    assert restore(tmp_path, tmp_path / "state.json") == []


def test_gpu_tuned_locks_the_clock_at_max_and_checks_it(tmp_path):
    smi, state = FakeSmi(), tmp_path / "state.json"
    set_group(tmp_path, "gpu", "tuned", state, run=smi)
    assert smi.changes == [["-lgc", "2520,2520"]] and smi.clock == 2520
    assert "gpu" in json.loads(state.read_text())


def test_gpu_lock_that_doesnt_hold_stops_with_didnt_take(tmp_path):
    with pytest.raises(NotTaken, match="210 MHz"):
        set_group(tmp_path, "gpu", "tuned", tmp_path / "state.json", run=FakeSmi(ignores=True))


def test_gpu_as_found_resets_the_clocks_and_restore_does_too(tmp_path):
    smi, state = FakeSmi(), tmp_path / "state.json"
    set_group(tmp_path, "gpu", "as-found", state, run=smi)
    assert smi.changes == []                      # nothing saved: the clocks were never touched
    set_group(tmp_path, "gpu", "tuned", state, run=smi)
    set_group(tmp_path, "gpu", "as-found", state, run=smi)
    assert smi.changes[-1] == ["-rgc"] and smi.clock == 210
    set_group(tmp_path, "gpu", "tuned", state, run=smi)
    assert restore(tmp_path, state, run=smi) == ["gpu"]
    assert smi.changes[-1] == ["-rgc"] and smi.clock == 210 and not state.exists()


def test_gpu_group_without_nvidia_smi_cant_run_here(tmp_path):
    with pytest.raises(CantRunHere, match="nvidia-smi"):
        set_group(tmp_path, "gpu", "tuned", tmp_path / "state.json", run=no_gpu)


def args(tmp_path, *words):
    return [*words, "--root", str(tmp_path), "--state", str(tmp_path / "state.json")]


def test_dry_run_prints_the_change_and_touches_nothing(tmp_path, capsys):
    make_machine(tmp_path)
    before = check(tmp_path, run=no_gpu)
    main(args(tmp_path, "memory", "tuned", "--dry-run"), run=no_gpu)
    out = capsys.readouterr().out
    assert "swappiness 60 -> 10" in out and "thp madvise -> always" in out
    assert check(tmp_path, run=no_gpu) == before and not (tmp_path / "state.json").exists()


def test_command_line_exit_codes(tmp_path, monkeypatch, capsys):
    """2 = can't run here, 1 = didn't take: either stops the A/B tool before a pointless round."""
    make_machine(tmp_path)
    with pytest.raises(SystemExit) as e:
        main(args(tmp_path, "gpu", "tuned"), run=no_gpu)
    assert e.value.code == 2
    monkeypatch.setattr(tune, "_write", lambda path, text: None)
    with pytest.raises(SystemExit) as e:
        main(args(tmp_path, "memory", "tuned"), run=no_gpu)
    assert e.value.code == 1 and "DIDN'T TAKE" in capsys.readouterr().out


def test_check_and_restore_from_the_command_line(tmp_path, capsys):
    make_machine(tmp_path)
    main(args(tmp_path, "check"), run=no_gpu)
    out = capsys.readouterr().out
    assert "memory: available (thp madvise, swappiness 60)" in out and "gpu: can't run here" in out
    main(args(tmp_path, "network", "tuned"), run=no_gpu)
    main(args(tmp_path, "restore"), run=no_gpu)
    assert "restored: network" in capsys.readouterr().out


def test_check_shows_gpu_power_and_slowdown_reasons(tmp_path):
    """Clock alone can't tell a lock that held from one the power limit pulled down."""
    make_machine(tmp_path)
    g = check(tmp_path, run=FakeSmi())["gpu"]
    assert g["available"] and g["values"]["sm_clock_mhz"] == 210
    assert g["status"] == {"power_w": 61.3, "power_cap": "Not Active", "hw_slowdown": "Not Active",
                           "heat_slowdown": "Not Active"}


def test_a_status_field_this_driver_lacks_doesnt_block_the_gpu_group(tmp_path):
    make_machine(tmp_path)
    smi = FakeSmi()
    smi.no_status = True
    g = check(tmp_path, run=smi)["gpu"]
    assert g["available"] and "not a valid field" in g["status"]["unavailable"]


def test_check_notes_when_there_is_no_swap(tmp_path):
    """Without swap, swappiness has nothing to act on: the memory A/B is then huge pages alone."""
    make_machine(tmp_path)
    (tmp_path / "proc/meminfo").write_text("MemTotal:       98000000 kB\nSwapTotal:             0 kB\n")
    m = check(tmp_path, run=no_gpu)["memory"]
    assert m["values"] == {"thp": "madvise", "swappiness": "60"}      # saved/restored values unchanged
    assert m["note"] == "no swap: swappiness has nothing to act on"
    (tmp_path / "proc/meminfo").write_text("SwapTotal:       4194304 kB\n")
    assert check(tmp_path, run=no_gpu)["memory"]["note"] == "swap 4096 MB"


def test_restore_keeps_going_past_a_failure_and_keeps_the_originals(tmp_path, monkeypatch):
    """A memory write that doesn't take must not leave the GPU clock locked."""
    make_machine(tmp_path)
    smi, state = FakeSmi(), tmp_path / "state.json"
    set_group(tmp_path, "memory", "tuned", state, run=smi)
    set_group(tmp_path, "gpu", "tuned", state, run=smi)
    monkeypatch.setattr(tune, "_write", lambda path, text: None)
    with pytest.raises(NotTaken, match="memory"):
        restore(tmp_path, state, run=smi)
    assert smi.changes[-1] == ["-rgc"] and smi.clock == 210      # the GPU was still put back
    assert json.loads(state.read_text())["memory"] == {"thp": "madvise", "swappiness": "60"}  # kept: retry


def test_restore_reports_a_failing_gpu_reset_after_the_other_groups(tmp_path):
    make_machine(tmp_path)
    smi, state = FakeSmi(), tmp_path / "state.json"
    set_group(tmp_path, "network", "tuned", state, run=smi)
    set_group(tmp_path, "gpu", "tuned", state, run=smi)
    smi.fails = ("-rgc",)
    with pytest.raises(NotTaken, match="gpu: nvidia-smi -rgc failed"):
        restore(tmp_path, state, run=smi)
    assert check(tmp_path, run=no_gpu)["network"]["values"] == {"netdev_max_backlog": "1000"}
    assert state.exists()


def test_command_line_failures_are_messages_not_tracebacks(tmp_path, monkeypatch, capsys):
    make_machine(tmp_path)
    smi = FakeSmi()
    smi.fails = ("-lgc",)
    with pytest.raises(SystemExit) as e:
        main(args(tmp_path, "gpu", "tuned"), run=smi)
    assert e.value.code == 1 and "nvidia-smi -lgc 2520,2520 failed" in capsys.readouterr().out
    smi.fails = ()
    main(args(tmp_path, "memory", "tuned"), run=smi)
    monkeypatch.setattr(tune, "_write", lambda path, text: None)
    with pytest.raises(SystemExit) as e:
        main(args(tmp_path, "restore"), run=smi)
    assert e.value.code == 1 and "DIDN'T TAKE" in capsys.readouterr().out


def test_check_warns_about_originals_left_by_an_earlier_session(tmp_path, capsys):
    """/var/tmp survives a node stop/start; an aborted session's saved originals mustn't be
    reused silently (a later "tuned" keeps whatever is saved)."""
    make_machine(tmp_path)
    main(args(tmp_path, "check"), run=no_gpu)
    assert "WARNING" not in capsys.readouterr().out
    main(args(tmp_path, "memory", "tuned"), run=no_gpu)
    capsys.readouterr()
    main(args(tmp_path, "check"), run=no_gpu)
    out = capsys.readouterr().out
    assert "WARNING: originals saved" in out and "(memory)" in out and "run restore" in out
