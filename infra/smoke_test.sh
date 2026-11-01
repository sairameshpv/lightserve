#!/usr/bin/env bash
# Is the platform really working? One PASS/FAIL line per check; exit code = number of failures.
# Runs against the current kubectl context: the real cluster after terraform apply, or kind.
#
#   infra/smoke_test.sh                  # every check
#   infra/smoke_test.sh --no-gpu mlflow  # kind: skip the GPU checks, run only some (names below)
#
# Checks: gpu lightserve prometheus grafana alert mlflow airflow traces logs chaos
# Each runs inside a pod that already has Python (lightserve, MLflow, the Airflow scheduler), so
# no extra images are needed. Secrets are read from Kubernetes and never printed.
set -uo pipefail
cd "$(dirname "$0")/.."
GPU=1
[[ ${1:-} == --no-gpu ]] && { GPU=0; shift; }
WANT=("$@")
FAILS=0
PROM=http://monitoring-kube-prometheus-prometheus.monitoring:9090

want() { [[ ${#WANT[@]} == 0 ]] && return 0; local w; for w in "${WANT[@]}"; do [[ $w == "$1" ]] && return 0; done; return 1; }
check() {  # check <name> <description> <command...>: PASS if the command exits 0
  local name=$1 desc=$2; shift 2
  want "$name" || return 0
  local out
  if out=$("$@" 2>&1); then echo "PASS  $name: $desc ${out:+($out)}"
  else echo "FAIL  $name: $desc"; echo "$out" | tail -5 | sed 's/^/        /'; FAILS=$((FAILS + 1)); fi
}
py() {  # py <namespace> <workload> <python code>: run Python inside that pod
  kubectl -n "$1" exec "$2" -c "${CONTAINER:-${2#*/}}" -- python -c "$3"
}
ls_py() { CONTAINER=lightserve py lightserve deploy/lightserve "$1"; }

gpu_node() {
  kubectl get nodes -l lightserve/pool=gpu -o jsonpath='{range .items[*]}{.status.conditions[?(@.type=="Ready")].status} {.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' |
    awk '$1 == "True" && $2 >= 1 {ok=1; print "Ready, " $2 " GPU allocatable"} END {exit !ok}'
}
lightserve() {  # health + one completion, from inside the pod
  ls_py 'import json, urllib.request as u
h = json.load(u.urlopen("http://localhost:8000/health", timeout=10))
req = u.Request("http://localhost:8000/v1/completions", data=json.dumps({"prompt": [1, 2, 3], "max_tokens": 8}).encode(),
                headers={"Content-Type": "application/json"})
c = json.load(u.urlopen(req, timeout=120))
assert h["status"] == "ok" and len(c["choices"][0]["token_ids"]) == 8, (h, c)
print("health ok, 8 tokens back")'
}
prometheus() {
  ls_py "import json, urllib.request as u, urllib.parse as p
r = json.load(u.urlopen('$PROM/api/v1/query?' + p.urlencode({'query': 'up{job=\"lightserve\"}'}), timeout=10))['data']['result']
assert r and r[0]['value'][1] == '1', r
g = [x['name'] for x in json.load(u.urlopen('$PROM/api/v1/rules', timeout=10))['data']['groups']]
assert {'lightserve', 'canary', 'cost'} <= set(g), g
print('lightserve target up; repo rule groups loaded')"
}
grafana() {
  local user pass
  user=$(kubectl -n monitoring get secret grafana-admin -o jsonpath='{.data.admin-user}' | base64 -d)
  pass=$(kubectl -n monitoring get secret grafana-admin -o jsonpath='{.data.admin-password}' | base64 -d)
  ls_py "import base64, json, urllib.request as u, sys
auth = base64.b64encode(sys.stdin.readline().strip().encode()).decode()
titles = []
for uid in ('lightserve', 'cost'):
    r = u.Request('http://monitoring-grafana.monitoring/api/dashboards/uid/' + uid, headers={'Authorization': 'Basic ' + auth})
    titles.append(json.load(u.urlopen(r, timeout=10))['dashboard']['title'])
print('dashboards: ' + ', '.join(titles))" <<<"$user:$pass"
}
alert() {  # an alert sent to Alertmanager arrives as an email in MailHog
  ls_py "import json, time, urllib.request as u
a = [{'labels': {'alertname': 'SmokeTest', 'severity': 'ticket'}, 'annotations': {'summary': 'smoke test'}}]
u.urlopen(u.Request('http://monitoring-kube-prometheus-alertmanager.monitoring:9093/api/v2/alerts', data=json.dumps(a).encode(),
                    headers={'Content-Type': 'application/json'}), timeout=10)
for _ in range(30):
    subjects = [m['Content']['Headers']['Subject'][0] for m in json.load(u.urlopen('http://mailhog.monitoring:8025/api/v2/messages', timeout=10))['items']]
    if any('SmokeTest' in s for s in subjects):
        print('email arrived'); break
    time.sleep(2)
else:
    raise SystemExit('no SmokeTest email in MailHog after 60 s')"
}
mlflow() {  # log a run with an artifact, register a model version, from inside MLflow's pod
  CONTAINER=mlflow py mlops deploy/mlflow 'import os, tempfile, time, mlflow
mlflow.set_tracking_uri("http://mlflow.mlops:80")
mlflow.set_experiment("smoke")
with mlflow.start_run() as run:
    mlflow.log_metric("x", 1.0)
    p = os.path.join(tempfile.mkdtemp(), "note.txt"); open(p, "w").write("hello"); mlflow.log_artifact(p)
c = mlflow.MlflowClient()
name = "smoke-%d" % time.time()
c.create_registered_model(name)
v = c.create_model_version(name, source=run.info.artifact_uri, run_id=run.info.run_id)
assert [a.path for a in c.list_artifacts(run.info.run_id)] == ["note.txt"]
print("run logged, artifact stored, %s version %s" % (name, v.version))' 2>&1 | tail -1  # MLflow's client chats (no git in its pod)
}
airflow() {  # the cluster's Airflow parsed the DAG from GitHub; the whole-loop test runs it in dry-run
  # in a pod of its own (same image, own memory: inside the scheduler it was OOM-killed)
  local errs image
  errs=$(kubectl -n mlops exec airflow-scheduler-0 -c scheduler -- airflow dags list-import-errors -o plain 2>/dev/null | grep -c drift_retrain)
  [[ $errs == 0 ]] || { echo "drift_retrain has import errors"; return 1; }
  kubectl -n mlops exec airflow-scheduler-0 -c scheduler -- airflow dags list -o plain 2>/dev/null | grep -q drift_retrain ||
    { echo "drift_retrain not listed"; return 1; }
  image=$(kubectl -n mlops get pod airflow-scheduler-0 -o jsonpath='{.spec.containers[?(@.name=="scheduler")].image}')
  kubectl -n mlops delete pod loop-test --ignore-not-found --wait >/dev/null
  kubectl -n mlops run loop-test --rm -i --restart=Never --pod-running-timeout=15m --image="$image" --image-pull-policy=IfNotPresent \
    --overrides='{"spec": {"containers": [{"name": "loop-test", "image": "'"$image"'", "resources": {"requests": {"memory": "1Gi"}, "limits": {"memory": "2560Mi"}},
      "command": ["bash", "-c", "git clone -q --depth 1 https://github.com/sairameshpv/lightserve.git /tmp/repo && cd /tmp/repo && python -m pytest mlops/tests/test_drift_loop.py -q -p no:cacheprovider 2>&1 | tail -1"]}]}}' \
    2>/dev/null | grep -E "^[0-9]+ passed" | grep -v failed | sed 's/^/DAG listed, no import errors; whole-loop test in the Airflow image: /'
}
traces() {  # requests traced to Tempo (10% sampled: send 40)
  ls_py 'import json, time, urllib.request as u
for _ in range(40):
    u.urlopen(u.Request("http://localhost:8000/v1/completions", data=json.dumps({"prompt": [1, 2], "max_tokens": 4}).encode(),
                        headers={"Content-Type": "application/json"}), timeout=60).read()
for _ in range(30):
    t = json.load(u.urlopen("http://tempo.monitoring:3200/api/search?tags=service.name%3Dlightserve&limit=5", timeout=10)).get("traces", [])
    if t:
        print("%d trace(s), e.g. %s" % (len(t), t[0]["rootTraceName"])); break
    time.sleep(2)
else:
    raise SystemExit("no lightserve trace in Tempo after 60 s")'
}
logs() {  # request log lines shipped by Alloy to Loki
  ls_py 'import json, time, urllib.request as u, urllib.parse as p
q = p.urlencode({"query": "{app=\"lightserve\"} |= \"request_id\"", "limit": 1})
for _ in range(30):
    r = json.load(u.urlopen("http://loki.monitoring:3100/loki/api/v1/query_range?" + q, timeout=10))["data"]["result"]
    if r:
        print("request log line in Loki"); break
    time.sleep(2)
else:
    raise SystemExit("no lightserve request log in Loki after 60 s")'
}
chaos() {
  kubectl -n chaos-mesh get pods --no-headers | awk '$3 == "Running" {n++} END {print n " Chaos Mesh pods running"; exit !(n >= 2)}'
}

[[ $GPU == 1 ]] && check gpu "GPU node Ready with a GPU Kubernetes can schedule" gpu_node
check lightserve "lightserve answers /health and a completion" lightserve
check prometheus "Prometheus scrapes lightserve; the repo's alert rules are loaded" prometheus
check grafana "Grafana serves the lightserve and Cost dashboards" grafana
check alert "an alert reaches MailHog as an email" alert
check mlflow "MLflow logs a run, stores an artifact, registers a model" mlflow
check airflow "Airflow runs the retraining DAG (dry-run, fakes)" airflow
check traces "a request's trace in Tempo" traces
check logs "a request's log line in Loki" logs
check chaos "Chaos Mesh running" chaos
echo "$FAILS failed"
exit "$FAILS"
