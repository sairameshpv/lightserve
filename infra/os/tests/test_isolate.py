"""Tests for infra/os/isolate.py: the exact commands for each mode, from the topology report's
layout (the fake L40S-shaped VM: housekeeping 0,12; engine 1, twin 13 idle; server 2,14; rest the
other 18), and moving interrupts on a fake machine folder, then putting them back exactly."""
import json

from infra.os.isolate import STOP, Config, check_pinning, main, parse_task_status, plan, restore_irqs, steer_irqs
from test_topology import make_machine

LAYOUT = {"housekeeping": [0, 12], "engine": 1, "engine_twin_idle": [13], "server": [2, 14],
          "rest": [c for c in range(24) if c not in (0, 12, 1, 13, 2, 14)]}
ALL = list(range(24))
CFG = Config(image="ghcr.io/sairameshpv/lightserve:gpu-latest", gpus=True, server_args="--n-layers 4 --max-model-len 512")


def runs(p, name):
    return [c for c in p["commands"] if c.startswith("docker run") and f"--name {name} " in c]


def test_quiet_is_lightserve_alone_unpinned():
    p = plan("quiet", LAYOUT, ALL, CFG)
    assert p["commands"][0] == STOP
    [ls] = runs(p, "lightserve")
    assert "--cpuset-cpus" not in ls and "--engine-cpus" not in ls
    assert "--gpus all" in ls and ls.endswith("ghcr.io/sairameshpv/lightserve:gpu-latest --n-layers 4 --max-model-len 512")
    assert runs(p, "noise") == [] and p["irqs"] == "restore"


def test_noisy_burns_every_core_but_housekeeping():
    """Housekeeping (0,12) belongs to the OS and the load generator in every state: if the
    neighbour slowed the thing measuring lightserve, the A/B would mix the two up."""
    p = plan("noisy", LAYOUT, ALL, CFG)
    [noise] = runs(p, "noise")
    assert "--cpuset-cpus 1-11,13-23 " in noise and "seq 22" in noise
    [ls] = runs(p, "lightserve")
    assert "--cpuset-cpus" not in ls and p["irqs"] == "restore"


def test_isolated_noisy_keeps_the_neighbour_off_lightserves_cores():
    p = plan("isolated-noisy", LAYOUT, ALL, CFG)
    [noise] = runs(p, "noise")
    assert "--cpuset-cpus 3-11,15-23 " in noise and "seq 18" in noise
    [ls] = runs(p, "lightserve")
    assert "--cpuset-cpus 1-2,14 " in ls          # engine + server cores; the engine's twin (13) stays idle
    assert ls.endswith("--n-layers 4 --max-model-len 512 --engine-cpus 1 --server-cpus 2,14")
    assert p["irqs"] == "steer"


def test_isolated_quiet_is_pinned_without_a_neighbour():
    p = plan("isolated-quiet", LAYOUT, ALL, CFG)
    assert runs(p, "noise") == [] and "--engine-cpus 1" in runs(p, "lightserve")[0] and p["irqs"] == "steer"


def test_restore_stops_everything_and_puts_interrupts_back():
    p = plan("restore", LAYOUT, ALL, CFG)
    assert p["commands"] == [STOP] and p["irqs"] == "restore"


def test_mac_rehearsal_config_has_no_gpu():
    p = plan("quiet", LAYOUT, ALL, Config(image="lightserve:fake", gpus=False, server_args="--fake-engine"))
    [ls] = runs(p, "lightserve")
    assert "--gpus" not in ls and ls.endswith("lightserve:fake --fake-engine")


# --- moving interrupts ---

def test_steer_moves_interrupts_off_reserved_cores_and_restore_puts_them_back(tmp_path):
    make_machine(tmp_path, n_cpus=24)          # irq 0 on cpu 0, 24 (nvidia) on 0-23, 25 (virtio) on 1
    state = tmp_path / "isolate_state.json"
    report = steer_irqs(tmp_path, LAYOUT, state, irqbalance_running=True)
    assert (tmp_path / "proc/irq/24/smp_affinity_list").read_text().strip() == "0,12"
    assert (tmp_path / "proc/irq/25/smp_affinity_list").read_text().strip() == "0,12"
    assert (tmp_path / "proc/irq/0/smp_affinity_list").read_text().strip() == "0"   # already on housekeeping
    assert report["moved"] == [24, 25] and report["failed"] == []
    saved = json.loads(state.read_text())
    assert saved == {"irqs": {"24": "0-23", "25": "1"}, "irqbalance_was_running": True}
    assert restore_irqs(tmp_path, state) == {"restored": [24, 25], "start_irqbalance": True}
    assert (tmp_path / "proc/irq/24/smp_affinity_list").read_text().strip() == "0-23"
    assert (tmp_path / "proc/irq/25/smp_affinity_list").read_text().strip() == "1"
    assert not state.exists()


def test_steering_twice_keeps_the_true_originals(tmp_path):
    """The A/B alternates rounds; a second steer must not save the already-steered values."""
    make_machine(tmp_path, n_cpus=24)
    state = tmp_path / "isolate_state.json"
    steer_irqs(tmp_path, LAYOUT, state, irqbalance_running=True)
    steer_irqs(tmp_path, LAYOUT, state, irqbalance_running=False)
    assert json.loads(state.read_text()) == {"irqs": {"24": "0-23", "25": "1"}, "irqbalance_was_running": True}


def test_restore_without_saved_state_does_nothing(tmp_path):
    make_machine(tmp_path, n_cpus=24)
    assert restore_irqs(tmp_path, tmp_path / "none.json") == {"restored": [], "start_irqbalance": False}


def test_an_interrupt_linux_wont_move_is_reported_not_fatal(tmp_path):
    """Some interrupts are kernel-managed: writing their cores fails. Note it, move the rest."""
    make_machine(tmp_path, n_cpus=24)
    (tmp_path / "proc/irq/24/smp_affinity_list").chmod(0o444)  # readable, but a write fails, as Linux does
    report = steer_irqs(tmp_path, LAYOUT, tmp_path / "s.json", irqbalance_running=False)
    assert report["moved"] == [25] and report["failed"] == [24]



# --- did it take effect? ---

def status_dump(threads):
    """What `docker exec lightserve sh -c 'for t in /proc/1/task/*; do echo "== $t"; cat $t/status; done'` prints."""
    return "".join(f"== /proc/1/task/{tid}\nName:\tpython\nCpus_allowed:\tffffff\nCpus_allowed_list:\t{cpus}\n"
                   for tid, cpus in threads)


def test_parse_task_status():
    assert parse_task_status(status_dump([(1, "2,14"), (7, "1")])) == {1: [2, 14], 7: [1]}


def test_pinning_ok_one_engine_thread_rest_on_server_cores():
    r = check_pinning(parse_task_status(status_dump([(1, "2,14"), (7, "1"), (9, "2,14")])), LAYOUT)
    assert r["ok"] is True


def test_pinning_wrong_when_no_thread_sits_on_the_engine_core():
    """e.g. an image without --engine-cpus support: every thread on the container's cores."""
    r = check_pinning(parse_task_status(status_dump([(1, "1-2,14"), (7, "1-2,14")])), LAYOUT)
    assert r["ok"] is False and "engine" in r["why"]


def test_pinning_wrong_when_another_thread_shares_the_engine_core():
    r = check_pinning(parse_task_status(status_dump([(1, "2,14"), (7, "1"), (8, "1")])), LAYOUT)
    assert r["ok"] is False


def test_dry_run_prints_the_commands_and_runs_nothing(tmp_path, capsys):
    topo = tmp_path / "topology.json"
    topo.write_text(json.dumps({"facts": {"cpus": ALL}, "advice": {"layout": LAYOUT}}))
    main(["isolated-noisy", "--topology", str(topo), "--dry-run"])
    out = capsys.readouterr().out
    assert "docker rm -f lightserve noise" in out and "--cpuset-cpus 3-11,15-23" in out
    assert "--engine-cpus 1 --server-cpus 2,14" in out and "interrupts: steer to 0,12" in out
