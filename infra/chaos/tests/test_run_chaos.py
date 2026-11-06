"""Tests for the judges in infra/chaos/run_chaos.py: each experiment's pass rule, written before
it runs, applied to made-up request records, phase times and facts from Prometheus/Kubernetes."""
from infra.chaos.run_chaos import JUDGES

PHASES = {"normal": (0, 30), "fault": (30, 120), "recover": (120, 180)}


def rec(t, outcome="ok", ttft=0.05, total=0.7):
    return {"t_start": t, "outcome": outcome, "ttft_s": ttft, "total_s": total}


def steady(phase, **kw):
    a, b = PHASES[phase]
    return [rec(t, **kw) for t in range(a, b, 2)]


def judge(name, records, **facts):
    return JUDGES[name](records, PHASES, facts)


# --- rules shared by every experiment with traffic ---

def test_one_hung_request_means_not_healed_even_if_everything_else_recovered():
    records = steady("normal") + steady("fault", outcome="error") + steady("recover")
    records.append(rec(60, outcome="hung"))
    r = judge("pod-kill", records, alert_states=["pending"], pod_ready=True, in_flight_end=0)
    assert r["caught"][0] and not r["healed"][0] and "hung" in r["healed"][1] and not r["pass"]


def test_a_phase_without_requests_cant_be_judged():
    r = judge("network-delay", steady("normal") + steady("recover"))
    assert not r["pass"] and "no requests" in r["caught"][1]


# --- 1 pod kill ---

def test_pod_kill_passes_when_the_outage_was_seen_and_serving_came_back():
    records = steady("normal") + steady("fault", outcome="error") + steady("recover")
    r = judge("pod-kill", records, alert_states=[], pod_ready=True, in_flight_end=0)
    assert r["pass"], r   # the load saw it even though the alert never left "inactive" (new pod too quick)


def test_pod_kill_fails_if_recovery_still_has_failures():
    records = steady("normal") + steady("fault", outcome="error") + steady("recover", outcome="error")
    r = judge("pod-kill", records, alert_states=["firing"], pod_ready=True, in_flight_end=0)
    assert not r["healed"][0] and "recovery" in r["healed"][1]


# --- 2 network delay ---

def test_network_delay_caught_by_slower_first_tokens_and_healed_when_back():
    records = steady("normal") + steady("fault", ttft=0.16) + steady("recover", ttft=0.055)
    assert judge("network-delay", records)["pass"]


def test_network_delay_too_small_a_rise_is_not_caught():
    records = steady("normal") + steady("fault", ttft=0.09) + steady("recover")
    r = judge("network-delay", records)
    assert not r["caught"][0] and "80 ms" in r["caught"][1]


# --- 3 queue flood ---

def test_flood_passes_when_the_server_was_full_endings_clean_and_no_restart():
    """Caught = the server hit its batch limit (256 at once). The first rule, "waiting queue > 0",
    couldn't see the kind rehearsal's real ~50 ms queue: Prometheus reads that gauge every 5 s."""
    records = steady("normal") + steady("fault", outcome="aborted", total=60) + steady("recover")
    r = judge("queue-flood", records, max_running=256, batch_limit=256, waiting_end=0, restarts=0, oom=False)
    assert r["pass"], r   # timed out at the server's 60 s with a proper ending counts as clean


def test_flood_that_never_filled_the_server_is_not_caught():
    records = steady("normal") + steady("fault") + steady("recover")
    r = judge("queue-flood", records, max_running=40, batch_limit=256, waiting_end=0, restarts=0, oom=False)
    assert not r["caught"][0] and "256" in r["caught"][1]


def test_flood_fails_on_an_out_of_memory_kill():
    records = steady("normal") + steady("fault") + steady("recover")
    r = judge("queue-flood", records, max_running=256, batch_limit=256, waiting_end=0, restarts=1, oom=True)
    assert not r["healed"][0] and "memory" in r["healed"][1]


# --- 4 noisy neighbour ---

def test_neighbour_caught_by_a_slower_p99_and_healed_when_it_returns():
    records = steady("normal") + steady("fault", total=1.0) + steady("recover", total=0.72)
    assert judge("noisy-neighbour", records)["pass"]


def test_neighbour_with_no_visible_effect_is_not_caught():
    records = steady("normal") + steady("fault", total=0.72) + steady("recover")
    r = judge("noisy-neighbour", records)
    assert not r["caught"][0] and "10%" in r["caught"][1]


# --- 5 drain (one written human step) ---

def test_drain_passes_with_the_alert_the_human_step_and_a_ready_pod():
    records = steady("normal") + steady("fault", outcome="error") + steady("recover")
    r = judge("drain", records, alert_states=["pending", "firing"], firing_after_s=45, human_step_logged=True,
              ready_after_uncordon_s=20, ready_limit_s=180, alert_cleared=True)
    assert r["pass"], r


def test_drain_fails_if_the_down_alert_never_fired():
    records = steady("normal") + steady("fault", outcome="error") + steady("recover")
    r = judge("drain", records, alert_states=["pending"], firing_after_s=None, human_step_logged=True,
              ready_after_uncordon_s=20, ready_limit_s=180, alert_cleared=True)
    assert not r["caught"][0] and "firing" in r["caught"][1]


# --- 6 MLflow down (no traffic: the registration is the workload) ---

def test_mlflow_down_passes_with_exactly_one_complete_champion_version():
    r = judge("mlflow-down", [], first_try_failed_or_waited=True, versions=1, champion_is_new=True, facts_complete=True)
    assert r["pass"], r


def test_mlflow_down_fails_on_a_half_registered_model():
    r = judge("mlflow-down", [], first_try_failed_or_waited=True, versions=1, champion_is_new=False, facts_complete=True)
    assert not r["healed"][0] and "champion" in r["healed"][1]


def test_percentile_is_nearest_rank():
    from infra.chaos.run_chaos import _pct
    values = list(range(1, 101))
    assert (_pct(values, 50), _pct(values, 99), _pct(values, 100)) == (50, 99, 100)
    assert (_pct([5], 99), _pct([3, 1, 2], 50)) == (5, 2)
    assert _pct(list(range(1, 11)), 99) == 10      # 10 values: the slowest one is the p99


# --- the runner: what it does, in order (--dry-run), and reading the answers it gets back ---

def dry(name, capsys, **kw):
    from infra.chaos.run_chaos import main
    main([name, "--dry-run", "--out", "/tmp/unused"] + [f"--{k.replace('_', '-')}={v}" for k, v in kw.items()])
    return capsys.readouterr().out.splitlines()


def order(lines, *needles):
    """Each needle appears, in this order (two may share a line: a command and its flag)."""
    at = 0
    for n in needles:
        hits = [i for i, line in enumerate(lines) if n in line and i >= at]
        assert hits, f"{n!r} missing at or after line {at}:\n" + "\n".join(lines)
        at = hits[0]


def test_every_traffic_experiment_ships_the_code_runs_load_then_fault_on_off_then_collects(capsys):
    lines = dry("pod-kill", capsys)
    order(lines, "create configmap chaos-code", "run chaos-load", "wait --for=condition=Ready pod/chaos-load",
          "apply -f infra/chaos/pod-kill.yaml", "delete -f infra/chaos/pod-kill.yaml",
          "logs chaos-load", "rollout status deploy/lightserve")
    assert any("--users 4" in line for line in lines)


def test_drain_logs_the_human_step_before_the_uncordon(capsys):
    lines = dry("drain", capsys)
    order(lines, "run chaos-load", "kubectl drain", "--pod-selector app=lightserve", "HUMAN STEP",
          "kubectl uncordon", "rollout status deploy/lightserve")


def test_flood_adds_more_users_than_one_batch_holds_for_the_fault_only(capsys):
    """lightserve batches up to 256 requests at once (--max-num-seqs): the kind rehearsal's 40
    users never made anything wait. 300 must queue."""
    lines = dry("queue-flood", capsys)
    order(lines, "run chaos-load ", "run chaos-flood", "--users 300 --duration-s 90",
          "logs chaos-flood", "delete pod chaos-flood", "logs chaos-load")   # stopped AT fault end (kind: it ran on)
    assert any("chaos-flood" in line and '"memory": "512Mi"' in line for line in lines)


def test_the_neighbour_pod_is_placed_before_the_normal_phase_and_burns_only_in_the_fault(capsys):
    lines = dry("noisy-neighbour", capsys)
    order(lines, "noisy-neighbour.yaml -l chaos-part=setup", "rollout status deploy/chaos-neighbour",
          "run chaos-load", "apply -f infra/chaos/noisy-neighbour.yaml -l chaos-part=fault",
          "delete -f infra/chaos/noisy-neighbour.yaml -l chaos-part=fault",   # AT fault end, not Chaos Mesh's own timer
          "logs chaos-load", "delete -f infra/chaos/noisy-neighbour.yaml --ignore-not-found")


def test_mlflow_down_starts_the_fault_then_the_registration_and_no_load(capsys):
    lines = dry("mlflow-down", capsys)
    order(lines, "create configmap chaos-code", "apply -f infra/chaos/mlflow-down.yaml", "run chaos-register",
          "register_probe.py", "logs chaos-register")
    assert not any("run chaos-load" in line for line in lines)


def test_phase_lengths_are_settings(capsys):
    lines = dry("network-delay", capsys, normal_s=5, fault_s=10, recover_s=5)
    assert any("--duration-s 35" in line for line in lines)   # 5 + 10 + 5 + 15 s of start-up slack


def test_alert_states_seen_from_a_prometheus_answer():
    from infra.chaos.run_chaos import alert_states, first_seen_after
    answer = {"data": {"result": [{"metric": {"alertname": "LightserveDown", "alertstate": "pending"}},
                                  {"metric": {"alertname": "LightserveDown", "alertstate": "firing"}}]}}
    assert alert_states(answer) == ["firing", "pending"]
    assert alert_states({"data": {"result": []}}) == []
    series = {"data": {"result": [{"values": [[1030.0, "1"], [1035.0, "1"]]}]}}
    assert first_seen_after(series, 1000.0) == 30.0 and first_seen_after({"data": {"result": []}}, 1000.0) is None


def test_restarts_and_out_of_memory_from_the_pod_status():
    from infra.chaos.run_chaos import pod_health
    before = {"items": [{"metadata": {"name": "ls-a"}, "status": {"containerStatuses": [{"restartCount": 0, "lastState": {}}]}}]}
    oom = {"items": [{"metadata": {"name": "ls-a"}, "status": {"containerStatuses": [
        {"restartCount": 1, "lastState": {"terminated": {"reason": "OOMKilled"}}}]}}]}
    assert pod_health(before, before) == {"restarts": 0, "oom": False}
    assert pod_health(before, oom) == {"restarts": 1, "oom": True}
    replaced = {"items": [{"metadata": {"name": "ls-b"}, "status": {"containerStatuses": [{"restartCount": 0, "lastState": {}}]}}]}
    assert pod_health(before, replaced)["restarts"] == 1    # a new pod is a restart too


def test_a_fault_command_that_prints_something_is_not_mistaken_for_saved_state():
    """In a real run `kubectl apply` prints "podchaos ... created"; the dry run prints nothing."""
    import io
    from pathlib import Path
    from infra.chaos.run_chaos import Cluster, Opts, traffic_experiment
    c = Cluster(Opts(out=Path("/tmp/unused"), dry_run=True), io.StringIO())
    records, phases, facts = traffic_experiment(
        c, "pod-kill", lambda img: "podchaos.chaos-mesh.org/lightserve-pod-kill created\n",
        lambda state: "podchaos.chaos-mesh.org \"lightserve-pod-kill\" deleted\n", lambda p, s: {"state": s})
    assert facts == {"state": {}} and set(phases) == {"normal", "fault", "recover"}


def test_every_label_the_runner_selects_by_exists_in_its_file(capsys):
    """kubectl apply -l X on a file with no object labelled X fails ("no objects passed to apply"):
    the kind rehearsal's noisy-neighbour run crashed on exactly that."""
    import re
    from pathlib import Path
    import yaml
    from infra.chaos.run_chaos import REPO
    for name in ("pod-kill", "network-delay", "queue-flood", "noisy-neighbour", "drain", "mlflow-down"):
        for line in dry(name, capsys):
            m = re.search(r"apply -f (\S+) -l (\S+)=(\S+)", line)
            if m:
                docs = list(yaml.safe_load_all(Path(REPO / m[1]).read_text()))
                labels = [d["metadata"].get("labels", {}) for d in docs]
                assert any(lb.get(m[2]) == m[3] for lb in labels), f"{m[1]} has no object labelled {m[2]}={m[3]}"


def test_a_crash_in_one_experiment_doesnt_hide_the_rows_already_judged(capsys, monkeypatch, tmp_path):
    import pytest
    import infra.chaos.run_chaos as rc
    ok = [rec(t) for t in range(0, 180, 2)]
    def judged(c):
        return ok, PHASES, {}
    def crashes(c):
        raise RuntimeError("kubectl went away")
    monkeypatch.setitem(rc.RUNNERS, "network-delay", judged)
    monkeypatch.setitem(rc.JUDGES, "network-delay", lambda r, p, f: {"caught": (True, "x"), "healed": (True, "y"), "pass": True})
    monkeypatch.setitem(rc.RUNNERS, "queue-flood", crashes)
    with pytest.raises(RuntimeError):
        rc.main(["network-delay", "queue-flood", "--out", str(tmp_path)])
    assert "PASS  network-delay" in capsys.readouterr().out


def test_the_registration_image_can_be_given_instead_of_asking_airflow(capsys):
    """kind: no Airflow installed just to read its image name; the real cluster asks Airflow."""
    lines = dry("mlflow-down", capsys)
    assert any("get pod airflow-scheduler-0" in line for line in lines)
    lines = dry("mlflow-down", capsys, register_image="airflow-mlops:local")
    assert not any("airflow-scheduler-0" in line for line in lines)
    assert any("run chaos-register --image=airflow-mlops:local" in line for line in lines)
