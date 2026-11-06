"""Chaos experiments (8j): break the platform on purpose, then judge whether the break was caught
and healed on its own, by rules fixed before the run (infra/README.md has the table).

This part: the judges. Each takes the load's request records (infra/chaos/load.py), the phase
times {"normal"|"fault"|"recover": (start, end)} and facts the runner read from Prometheus and
Kubernetes, and returns {"caught": (ok, why), "healed": (ok, why), "pass": both}.
"""
import math

NETWORK_RISE_S = 0.080     # first-token p50 must rise at least this much under +100 ms delay
NEIGHBOUR_RISE = 1.10      # p99 latency up at least 10% next to a CPU-burning neighbour
BACK_TO_NORMAL = 1.20      # "healed": within 20% of the normal phase
DRAIN_FIRING_S = 90        # LightserveDown firing within this long of the drain


def _pct(values, q):
    """Nearest-rank percentile (q in 0-100)."""
    v = sorted(values)
    return v[max(0, math.ceil(q / 100 * len(v)) - 1)]


def _phase(records, phases, name):
    a, b = phases[name]
    return [r for r in records if a <= r["t_start"] < b]


def _split(records, phases):
    """The three phases' records, or the name of the first empty one."""
    parts = {p: _phase(records, phases, p) for p in ("normal", "fault", "recover")}
    empty = [p for p, rs in parts.items() if not rs]
    return parts, (f"no requests in the {empty[0]} phase: can't judge" if empty else None)


def _hung(records):
    n = sum(r["outcome"] == "hung" for r in records)
    return f"{n} request(s) hung" if n else None


def _verdict(caught, healed):
    return {"caught": caught, "healed": healed, "pass": caught[0] and healed[0]}


def _cant(why):
    return _verdict((False, why), (False, why))


def _ms(s):
    return f"{s * 1000:.0f} ms"


def pod_kill(records, phases, facts):
    parts, empty = _split(records, phases)
    if empty:
        return _cant(empty)
    failed = sum(r["outcome"] != "ok" for r in parts["fault"])
    alert = sorted(set(facts["alert_states"]))
    caught = (failed > 0 or bool(alert),
              f"{failed} failed request(s) during the fault; LightserveDown {', '.join(alert) or 'never pending/firing'}")
    bad = sum(r["outcome"] != "ok" for r in parts["recover"])
    problems = [p for p in (_hung(records),
                            None if facts["pod_ready"] else "no Ready pod",
                            f"{bad} failed request(s) in recovery" if bad else None,
                            None if facts["in_flight_end"] == 0 else f"in-flight {facts['in_flight_end']} at the end")
                if p]
    return _verdict(caught, (not problems, "; ".join(problems) or "new pod Ready, recovery all ok, none hung"))


def network_delay(records, phases, facts):
    parts, empty = _split(records, phases)
    if empty:
        return _cant(empty)
    p50 = {p: _pct([r["ttft_s"] for r in rs if r["ttft_s"] is not None], 50) for p, rs in parts.items()}
    rise = p50["fault"] - p50["normal"]
    caught = (rise >= NETWORK_RISE_S, f"first-token p50 {_ms(p50['normal'])} -> {_ms(p50['fault'])} "
              f"(+{_ms(rise)}; needs +{_ms(NETWORK_RISE_S)})")
    problems = [p for p in (_hung(records),
                            None if p50["recover"] <= BACK_TO_NORMAL * p50["normal"]
                            else f"recovery p50 {_ms(p50['recover'])} not within 20% of normal") if p]
    return _verdict(caught, (not problems, "; ".join(problems) or f"recovery p50 {_ms(p50['recover'])}, none hung"))


def queue_flood(records, phases, facts):
    parts, empty = _split(records, phases)
    if empty:
        return _cant(empty)
    # The server full (its batch limit), not "waiting > 0": Prometheus reads gauges every 5 s and
    # the kind rehearsal's real queue lasted ~50 ms, so the waiting gauge read 0 (rule changed then).
    caught = (facts["max_running"] >= facts["batch_limit"],
              f"running peaked at {facts['max_running']:g} of the batch limit {facts['batch_limit']}")
    unclean = sum(r["outcome"] not in ("ok", "aborted") for r in records)
    p50 = {p: _pct([r["total_s"] for r in parts[p]], 50) for p in ("normal", "recover")}
    problems = [p for p in (_hung(records),
                            "out-of-memory kill" if facts["oom"] else None,
                            f"{facts['restarts']} restart(s)" if facts["restarts"] else None,
                            f"{unclean} request(s) ended with an error, not served or timed out" if unclean else None,
                            None if facts["waiting_end"] == 0 else f"queue still {facts['waiting_end']} at the end",
                            None if p50["recover"] <= BACK_TO_NORMAL * p50["normal"]
                            else "recovery latency not within 20% of normal") if p]
    return _verdict(caught, (not problems, "; ".join(problems) or "no restart, every request ended cleanly, queue back to 0"))


def noisy_neighbour(records, phases, facts):
    parts, empty = _split(records, phases)
    if empty:
        return _cant(empty)
    p99 = {p: _pct([r["total_s"] for r in rs], 99) for p, rs in parts.items()}
    caught = (p99["fault"] >= NEIGHBOUR_RISE * p99["normal"],
              f"p99 latency {_ms(p99['normal'])} -> {_ms(p99['fault'])} (needs +10%)")
    problems = [p for p in (_hung(records),
                            None if p99["recover"] <= BACK_TO_NORMAL * p99["normal"]
                            else f"recovery p99 {_ms(p99['recover'])} not within 20% of normal") if p]
    return _verdict(caught, (not problems, "; ".join(problems) or f"recovery p99 {_ms(p99['recover'])}, none hung"))


def drain(records, phases, facts):
    parts, empty = _split(records, phases)
    if empty:
        return _cant(empty)
    fired = "firing" in facts["alert_states"] and facts["firing_after_s"] is not None \
        and facts["firing_after_s"] <= DRAIN_FIRING_S
    caught = (fired, f"LightserveDown firing after {facts['firing_after_s']} s (needs firing within {DRAIN_FIRING_S} s)"
              if facts["firing_after_s"] is not None else "LightserveDown never firing")
    ready = facts["ready_after_uncordon_s"]
    problems = [p for p in (_hung(records),
                            None if facts["human_step_logged"] else "the human step (uncordon) wasn't logged",
                            None if ready is not None and ready <= facts["ready_limit_s"]
                            else f"no Ready pod within {facts['ready_limit_s']} s of the uncordon",
                            None if facts["alert_cleared"] else "LightserveDown didn't clear",
                            None if any(r["outcome"] == "ok" for r in parts["recover"]) else "no ok request in recovery")
                if p]
    return _verdict(caught, (not problems, "; ".join(problems) or f"uncordoned (human step), Ready {ready} s later, alert cleared"))


def mlflow_down(records, phases, facts):
    caught = (facts["first_try_failed_or_waited"], "the registration's first try failed or waited"
              if facts["first_try_failed_or_waited"] else "the first try went through as if MLflow were up")
    problems = [p for p in (None if facts["versions"] == 1 else f"{facts['versions']} new versions (want exactly 1)",
                            None if facts["champion_is_new"] else "the champion label isn't on the new version (half-registered)",
                            None if facts["facts_complete"] else "the new version is missing some of its facts")
                if p]
    return _verdict(caught, (not problems, "; ".join(problems) or "exactly one new version, champion, all facts"))


JUDGES = {"pod-kill": pod_kill, "network-delay": network_delay, "queue-flood": queue_flood,
          "noisy-neighbour": noisy_neighbour, "drain": drain, "mlflow-down": mlflow_down}


# --- the runner -------------------------------------------------------------------------------
# Run from a laptop whose kubectl points at the cluster (kind or Nebius). Each experiment: ship the
# code (a ConfigMap), steady load, 3 phases (normal / fault / recover), fault on/off, collect the
# records, read the facts, judge. Everything lands in --out/<experiment>/.

import argparse
import io
import json
import shlex
import subprocess
import sys
import tarfile
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CODE = ["infra/chaos/load.py", "infra/chaos/register_probe.py", *sorted(
    str(p.relative_to(REPO)) for p in (REPO / "training/qlora_financial").glob("*.py"))]
LS = "-n lightserve"
URL = "http://lightserve.lightserve:8000"
PROM = "/api/v1/namespaces/monitoring/services/monitoring-kube-prometheus-prometheus:9090/proxy/api/v1"
START_SLACK_S = 15           # the load runs this much longer than the 3 phases (pod start-up)
FAULT = {"pod-kill": "infra/chaos/pod-kill.yaml", "network-delay": "infra/chaos/network-delay.yaml",
         "noisy-neighbour": "infra/chaos/noisy-neighbour.yaml", "mlflow-down": "infra/chaos/mlflow-down.yaml"}


@dataclass
class Opts:
    out: Path
    normal_s: float = 30
    fault_s: float = 90
    recover_s: float = 60
    users: int = 4
    flood_users: int = 300   # above lightserve's batch of 256 (--max-num-seqs): 40 never queued (kind)
    ready_limit_s: float = 180   # drain: Ready again within this long of the uncordon (GPU: 300)
    batch_limit: int = 256       # lightserve's --max-num-seqs (its default; the chart doesn't set it)
    register_image: str = ""     # experiment 6's pod image; "" = Airflow's own (the real cluster)
    settle_s: float = 30         # quiet time before each experiment (kind: the neighbour ran slow right after the flood)
    dry_run: bool = False


def alert_states(answer):
    """A Prometheus answer about ALERTS -> the alert states seen ("pending", "firing")."""
    return sorted({r["metric"]["alertstate"] for r in answer["data"]["result"]})


def first_seen_after(series, t0):
    """A query_range answer -> seconds from t0 to its first sample (None if it never appeared)."""
    times = [float(v[0]) for r in series["data"]["result"] for v in r["values"]]
    return min(times) - t0 if times else None


def pod_health(before, after):
    """`kubectl get pods -o json` before and after -> restarts (a replaced pod counts) and OOM kills."""
    def state(doc):
        return {p["metadata"]["name"]: p["status"].get("containerStatuses", []) for p in doc["items"]}
    b, a = state(before), state(after)
    restarts = sum(c["restartCount"] for cs in a.values() for c in cs) - sum(
        c["restartCount"] for n, cs in b.items() if n in a for c in cs)
    restarts += sum(1 for n in b if n not in a)
    oom = any(c.get("lastState", {}).get("terminated", {}).get("reason") == "OOMKilled" for cs in a.values() for c in cs)
    return {"restarts": restarts, "oom": oom}


class Cluster:
    """kubectl and the clock; with dry_run it prints every command and waits for nothing."""

    def __init__(self, opts, log):
        self.opts, self.log = opts, log

    def sh(self, cmd, check=True):
        print("+ " + cmd, flush=True)
        self.log.write(f"{time.time():.1f} + {cmd}\n")
        if self.opts.dry_run:
            return ""
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if check and r.returncode:
            raise RuntimeError(f"{cmd}\n{r.stderr.strip()}")
        return r.stdout if r.returncode == 0 else None

    def note(self, text):
        print(text, flush=True)
        self.log.write(f"{time.time():.1f} {text}\n")

    def sleep(self, s):
        if not self.opts.dry_run:
            time.sleep(s)

    def prom(self, path, **params):
        out = self.sh(f"kubectl get --raw {shlex.quote(PROM + '/' + path + '?' + urllib.parse.urlencode(params))}")
        return json.loads(out) if out else {"data": {"result": []}}

    def value(self, query, at):
        r = self.prom("query", query=query, time=at)["data"]["result"]
        return float(r[0]["value"][1]) if r else None

    def json(self, cmd):
        out = self.sh(cmd)
        return json.loads(out) if out else {"items": []}


def code_configmap():
    """The files the pods need, packed: the rehearsal runs today's code, not GitHub's."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in CODE:
            tar.add(REPO / f, arcname=f)
    path = Path("/tmp") / f"chaos-code-{time.time_ns()}.tgz"
    path.write_bytes(buf.getvalue())
    return path


def pod(name, ns, image, args, mem="256Mi"):
    """`kubectl run` for a pod that unpacks the code and runs one of its scripts."""
    unpack = "import tarfile; tarfile.open('/code/code.tgz').extractall('/tmp/c')"
    spec = {"spec": {"restartPolicy": "Never", "containers": [{
        "name": name, "image": image, "command": ["sh", "-c", f'python -c "{unpack}" && cd /tmp/c && {args}'],
        "resources": {"requests": {"cpu": "100m", "memory": mem}, "limits": {"memory": mem}},
        "volumeMounts": [{"name": "code", "mountPath": "/code"}]}],
        "volumes": [{"name": "code", "configMap": {"name": "chaos-code"}}]}}
    return (f"kubectl -n {ns} run {name} --image={image} --restart=Never --overrides={shlex.quote(json.dumps(spec))}")


def ship_code(c, ns):
    tgz = "/tmp/chaos-code.tgz" if c.opts.dry_run else code_configmap()
    c.sh(f"kubectl -n {ns} create configmap chaos-code --from-file=code.tgz={tgz} --dry-run=client -o yaml | kubectl apply -f -")


def load_pod(c, name, users, duration_s, image, mem="256Mi"):
    c.sh(f"kubectl {LS} delete pod {name} --ignore-not-found --wait")
    c.sh(pod(name, "lightserve", image, f"python infra/chaos/load.py --url {URL} --users {users} --duration-s {duration_s:g}",
             mem=mem))


def wait_done(c, ns, name, timeout_s, every_s=3):
    """Until the pod has succeeded OR failed (kubectl wait can only wait for one of them: kind's
    probe failed after 18 s and was waited on for 10 min). Returns its last phase."""
    phase = ""
    for _ in range(max(1, int(timeout_s / every_s))):
        phase = (c.sh(f"kubectl -n {ns} get pod {name} -o jsonpath={{.status.phase}}", check=False) or "").strip()
        if phase in ("Succeeded", "Failed") or c.opts.dry_run:
            break
        c.sleep(every_s)
    return phase


def collect(c, name, wait_s):
    wait_done(c, "lightserve", name, wait_s)
    out = c.sh(f"kubectl {LS} logs {name}") or ""
    c.sh(f"kubectl {LS} delete pod {name} --ignore-not-found")
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def traffic_experiment(c, name, on, off, facts_after):
    """The shared sequence: load up, normal, fault on, fault, fault off, recover, collect, facts."""
    o = c.opts
    image = c.sh(f"kubectl {LS} get deploy lightserve -o jsonpath='{{.spec.template.spec.containers[0].image}}'") or "IMAGE"
    ship_code(c, "lightserve")
    total = o.normal_s + o.fault_s + o.recover_s + START_SLACK_S
    load_pod(c, "chaos-load", o.users, total, image)
    c.sh(f"kubectl {LS} wait --for=condition=Ready pod/chaos-load --timeout=120s")
    t0 = time.time()
    phases = {"normal": (t0, t0 + o.normal_s), "fault": (t0 + o.normal_s, t0 + o.normal_s + o.fault_s),
              "recover": (t0 + o.normal_s + o.fault_s, t0 + o.normal_s + o.fault_s + o.recover_s)}
    c.sleep(o.normal_s)
    state = on(image)
    state = state if isinstance(state, dict) else {}       # kubectl's own output is not state
    c.sleep(o.fault_s)
    more = off(state)
    state.update(more if isinstance(more, dict) else {})
    c.sleep(o.recover_s)
    records = collect(c, "chaos-load", START_SLACK_S + 120) + state.pop("extra_records", [])
    return records, phases, facts_after(phases, state)


def down_alert_facts(c, phases):
    a, end = phases["fault"][0], phases["recover"][1]
    seen = c.prom("query", query=f'max_over_time(ALERTS{{alertname="LightserveDown"}}[{end - a:.0f}s])', time=end)
    return {"alert_states": alert_states(seen)}


def run_pod_kill(c):
    f = FAULT["pod-kill"]
    def facts(phases, state):
        ok = c.sh(f"kubectl {LS} rollout status deploy/lightserve --timeout=120s", check=False) is not None
        c.sleep(15)          # one more scrape after the load ended
        return {**down_alert_facts(c, phases), "pod_ready": ok,
                "in_flight_end": c.value("sum(lightserve_requests_in_flight)", time.time())}
    return traffic_experiment(c, "pod-kill", lambda img: c.sh(f"kubectl apply -f {f}"),
                              lambda s: c.sh(f"kubectl delete -f {f} --ignore-not-found"), facts)


def run_simple(c, name):
    f = FAULT[name]
    return traffic_experiment(c, name, lambda img: c.sh(f"kubectl apply -f {f}"),
                              lambda s: c.sh(f"kubectl delete -f {f} --ignore-not-found"), lambda p, s: {})


def run_noisy_neighbour(c):
    f = FAULT["noisy-neighbour"]
    c.sh(f"kubectl apply -f {f} -l chaos-part=setup")             # placed before the normal phase
    c.sh(f"kubectl {LS} rollout status deploy/chaos-neighbour --timeout=120s")
    try:
        return traffic_experiment(c, "noisy-neighbour", lambda img: c.sh(f"kubectl apply -f {f} -l chaos-part=fault"),
                                  lambda s: c.sh(f"kubectl delete -f {f} -l chaos-part=fault --ignore-not-found"),
                                  lambda p, s: {})   # stopped AT fault end: Chaos Mesh's own timer starts late
    finally:
        c.sh(f"kubectl delete -f {f} --ignore-not-found")


def run_queue_flood(c):
    pods = f"kubectl {LS} get pods -l app=lightserve -o json"
    def on(image):
        before = c.json(pods)
        load_pod(c, "chaos-flood", c.opts.flood_users, c.opts.fault_s, image, mem="512Mi")   # 300 threads
        return {"before": before}
    def off(state):    # AT fault end: its own 90 s start late, and on kind it flooded the recovery
        out = c.sh(f"kubectl {LS} logs chaos-flood", check=False) or ""
        c.sh(f"kubectl {LS} delete pod chaos-flood --grace-period=0 --ignore-not-found")
        return {"extra_records": [json.loads(line) for line in out.splitlines() if line.startswith("{")]}
    def facts(phases, state):
        f0, f1 = phases["fault"]
        c.sleep(15)
        return {**pod_health(state["before"], c.json(pods)),
                "max_running": c.value(f"max(max_over_time(lightserve_requests_running[{f1 - f0 + 30:.0f}s]))", f1 + 30) or 0,
                "batch_limit": c.opts.batch_limit,
                "waiting_end": c.value("sum(lightserve_requests_waiting)", time.time())}
    return traffic_experiment(c, "queue-flood", on, off, facts)


def run_drain(c):
    node = c.sh(f"kubectl {LS} get pods -l app=lightserve -o jsonpath='{{.items[0].spec.nodeName}}'") or "NODE"
    def on(image):
        c.sh(f"kubectl drain {node} --pod-selector app=lightserve --ignore-daemonsets --delete-emptydir-data --timeout=90s")
    def off(state):
        c.note("HUMAN STEP: allow the node back (kubectl uncordon): one node, lightserve has nowhere else to go")
        c.sh(f"kubectl uncordon {node}")
        t = time.time()
        ok = c.sh(f"kubectl {LS} rollout status deploy/lightserve --timeout={c.opts.ready_limit_s:.0f}s", check=False)
        return {"human_step_logged": True, "ready_after_uncordon_s": round(time.time() - t, 1) if ok is not None else None}
    def facts(phases, state):
        a, end = phases["fault"][0], phases["recover"][1]
        firing = c.prom("query_range", query='ALERTS{alertname="LightserveDown",alertstate="firing"}', start=a, end=end, step=5)
        still = c.prom("query", query='ALERTS{alertname="LightserveDown",alertstate="firing"}', time=time.time())
        after = first_seen_after(firing, a)
        return {**down_alert_facts(c, phases), "firing_after_s": None if after is None else round(after, 1),
                "human_step_logged": state.get("human_step_logged", False),
                "ready_after_uncordon_s": state.get("ready_after_uncordon_s"), "ready_limit_s": c.opts.ready_limit_s,
                "alert_cleared": not still["data"]["result"]}
    try:
        return traffic_experiment(c, "drain", on, off, facts)
    finally:
        c.sh(f"kubectl uncordon {node}", check=False)                # never leave the node fenced off


def run_mlflow_down(c):
    f = FAULT["mlflow-down"]
    image = c.opts.register_image or c.sh(
        "kubectl -n mlops get pod airflow-scheduler-0 -o jsonpath='{.spec.containers[0].image}'") or "IMAGE"
    ship_code(c, "mlops")
    c.sh("kubectl -n mlops delete pod chaos-register --ignore-not-found --wait")
    c.sh(f"kubectl apply -f {f}")
    c.sleep(5)
    model = f"chaos-{int(time.time())}"
    c.sh(pod("chaos-register", "mlops", image, f"python infra/chaos/register_probe.py --tracking-uri http://mlflow.mlops:80 "
                                             f"--model {model} --retry-delay-s 60", mem="1Gi"))
    phase = wait_done(c, "mlops", "chaos-register", 600)
    out = c.sh("kubectl -n mlops logs chaos-register", check=False) or ""
    c.sh("kubectl -n mlops delete pod chaos-register --ignore-not-found")
    c.sh(f"kubectl delete -f {f} --ignore-not-found")
    lines = [line for line in out.splitlines() if line.startswith("{")]
    facts = json.loads(lines[-1]) if lines else {"first_try_failed_or_waited": False, "versions": 0,
                                                 "champion_is_new": False, "facts_complete": False}
    # kept either way: on kind the probe died with no answer and its error was thrown away with the pod
    return [], None, {**facts, "probe_phase": phase, "probe_log": out[-3000:]}


OWN_PODS = [("lightserve", "chaos-load"), ("lightserve", "chaos-flood"), ("mlops", "chaos-register")]
BASELINE_SLACK = 1.5    # a normal phase this much slower than the first experiment's is "baseline slow"


def cleanup(c):
    """The runner's own pods and every fault object, whether the experiment finished or crashed
    (kind: a crash left its load pod behind). Never lightserve or MLflow themselves."""
    for ns, name in OWN_PODS:
        c.sh(f"kubectl -n {ns} delete pod {name} --ignore-not-found --grace-period=0", check=False)
    for f in FAULT.values():
        c.sh(f"kubectl delete -f {f} --ignore-not-found", check=False)


def run_one(c, name, runner):
    try:
        return runner(c)
    finally:
        cleanup(c)


def check_no_leftovers(c):
    """Refuse to start while an earlier run's pods are still around (they'd load the cluster)."""
    left = [line.split("/", 1)[-1] for ns in ("lightserve", "mlops")
            for line in (c.sh(f"kubectl -n {ns} get pods -o name", check=False) or "").split()
            if line.split("/", 1)[-1].startswith("chaos-")]
    if left:
        raise RuntimeError(f"leftovers from an earlier run: {', '.join(left)} (delete them, then rerun)")


def _normal_p99(records, phases):
    a, b = phases["normal"]
    rs = [r["total_s"] for r in records if a <= r["t_start"] < b]
    return _pct(rs, 99) if rs else None


def baseline_check(records, phases, ref_p99):
    """This experiment's normal phase vs the first experiment's in the same run."""
    p99 = _normal_p99(records, phases)
    if ref_p99 is None:
        return True, "first experiment: it is the reference"
    if p99 is None:
        return False, "no requests in the normal phase"
    return (p99 <= BASELINE_SLACK * ref_p99,
            f"normal p99 {_ms(p99)} vs the first experiment's {_ms(ref_p99)} (allowed {BASELINE_SLACK}x)")


RUNNERS = {"pod-kill": run_pod_kill, "network-delay": lambda c: run_simple(c, "network-delay"),
           "queue-flood": run_queue_flood, "noisy-neighbour": run_noisy_neighbour, "drain": run_drain,
           "mlflow-down": run_mlflow_down}


def row(name, v):
    base = v.get("baseline", (True, ""))
    note = "" if base[0] else f" | BASELINE SLOW: {base[1]}"
    return f"{'PASS' if v['pass'] else 'FAIL'}  {name:16} caught: {v['caught'][1]} | healed: {v['healed'][1]}{note}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="+", choices=[*RUNNERS, "all"])
    ap.add_argument("--out", required=True, help="e.g. runs/chaos/kind")
    for k in ("normal_s", "fault_s", "recover_s", "ready_limit_s"):
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=getattr(Opts, k))
    ap.add_argument("--settle-s", type=float, default=Opts.settle_s, help="quiet time before each experiment")
    ap.add_argument("--register-image", default="", help="experiment 6's image (kind: airflow-mlops:local)")
    ap.add_argument("--dry-run", action="store_true", help="print every command, change nothing, judge nothing")
    a = ap.parse_args(argv)
    opts = Opts(out=Path(a.out), normal_s=a.normal_s, fault_s=a.fault_s, recover_s=a.recover_s,
                ready_limit_s=a.ready_limit_s, register_image=a.register_image, settle_s=a.settle_s,
                dry_run=a.dry_run)
    rows, ref = [], None
    for name in (list(RUNNERS) if "all" in a.experiments else a.experiments):
        out = opts.out / name
        if not opts.dry_run:
            out.mkdir(parents=True, exist_ok=True)
        log = io.StringIO() if opts.dry_run else open(out / "commands.log", "w")
        c = Cluster(opts, log)
        c.note(f"=== {name}")
        c.note(f"settle {opts.settle_s:g} s")
        c.sleep(opts.settle_s)
        check_no_leftovers(c)
        records, phases, facts = run_one(c, name, RUNNERS[name])
        if opts.dry_run:
            continue
        verdict = JUDGES[name](records, phases, facts)
        if phases is not None:   # experiments with traffic: is "normal" really normal?
            verdict["baseline"] = baseline_check(records, phases, ref)
            ref = ref if ref is not None else _normal_p99(records, phases)
            if not verdict["baseline"][0]:
                verdict["pass"] = False
        (out / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        (out / "facts.json").write_text(json.dumps({"phases": phases, **facts}, indent=2, default=str))
        (out / "verdict.json").write_text(json.dumps(verdict, indent=2))
        log.close()
        rows.append((name, verdict))
        print(row(name, verdict), flush=True)   # at once: a later crash mustn't hide it
    if len(rows) > 1:
        print("\n".join(["--- all"] + [row(n, v) for n, v in rows]))
    if rows and not all(v["pass"] for _, v in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
