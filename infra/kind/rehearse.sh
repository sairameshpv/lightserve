#!/usr/bin/env bash
# Rehearse the platform on a local kind cluster (no GPU, no cloud, free): the same charts,
# versions and settings as infra/terraform/helm.tf (both read infra/helm/releases.yaml), with
# lightserve on its fake engine and the GPU operator skipped.
#
#   infra/kind/rehearse.sh up                # create the kind cluster, load the fake lightserve image
#   infra/kind/rehearse.sh install <name>    # install one release from releases.yaml (or "all")
#   infra/kind/rehearse.sh uninstall <name>  # remove it again (Docker on a laptop can't hold all at once)
#   infra/kind/rehearse.sh down              # delete the kind cluster
set -euo pipefail
cd "$(dirname "$0")/../.."           # repo root
CLUSTER=lightserve
HELM_DIR=infra/helm

releases() {  # name namespace repo chart version values gpu wait, one line per release
  python3 -c '
import sys, yaml
for r in yaml.safe_load(open(sys.argv[1])):
    print(r["name"], r["namespace"], r.get("repo", "-"), r["chart"], r.get("version", "-"), r["values"],
          r.get("gpu", False), r.get("wait", True))
' "$HELM_DIR/releases.yaml"
}

extra_args() {  # what the Mac adds on top of the shared values
  case "$1" in
    platform)
      echo "--set-file alerts=monitoring/alerts.yml --set-file slo=monitoring/slo.yml" \
           "--set-file dashboards.lightserve=monitoring/grafana/provisioning/dashboards/lightserve.json" \
           "--set-file dashboards.cost=monitoring/grafana/provisioning/dashboards/cost.json" ;;
    lightserve)
      echo "--set engine=fake --set image.repository=lightserve --set image.tag=fake" ;;
    airflow)
      echo "--set images.airflow.repository=airflow-mlops --set images.airflow.tag=local --set images.airflow.pullPolicy=IfNotPresent" ;;
  esac
}

install() {
  local want=$1 found=0
  while read -r name ns repo chart version values gpu wait; do
    [[ $want != all && $want != "$name" ]] && continue
    found=1
    if [[ $gpu == True ]]; then echo "skip $name (GPU only)"; continue; fi
    kubectl create namespace "$ns" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
    local src=("$HELM_DIR/${chart#../}")
    [[ $repo != - ]] && src=("$chart" --repo "$repo" --version "$version")
    echo "install $name ($chart) into $ns"
    # shellcheck disable=SC2046
    local waitflag=(--wait --timeout 15m)
    [[ $wait == False ]] && waitflag=()
    helm upgrade --install "$name" "${src[@]}" -n "$ns" -f "$HELM_DIR/values/$values" \
      $(extra_args "$name") "${waitflag[@]}"
  done < <(releases)
  [[ $found == 1 ]] || { echo "no release named $want in releases.yaml" >&2; exit 1; }
}

case "${1:-}" in
  up)
    kind get clusters | grep -qx "$CLUSTER" || kind create cluster --name "$CLUSTER"
    kubectl config use-context "kind-$CLUSTER" >/dev/null
    docker image inspect lightserve:fake >/dev/null 2>&1 ||
      docker build -f infra/docker/lightserve.Dockerfile --target fake -t lightserve:fake .
    kind load docker-image lightserve:fake --name "$CLUSTER"
    docker image inspect airflow-mlops:local >/dev/null 2>&1 ||
      docker build -f infra/docker/airflow.Dockerfile -t airflow-mlops:local .
    kind load docker-image airflow-mlops:local --name "$CLUSTER"
    for ns in monitoring mlops lightserve chaos-mesh; do
      kubectl create namespace "$ns" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
    done
    # Grafana's admin secret, as Terraform makes it on the real cluster (a throwaway password here)
    kubectl -n monitoring create secret generic grafana-admin --from-literal=admin-user=admin \
      --from-literal=admin-password="$(openssl rand -hex 12)" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
    ;;
  install) install "${2:?release name or all}" ;;
  uninstall)
    ns=$(releases | awk -v n="${2:?release name}" '$1 == n {print $2}')
    helm uninstall "$2" -n "$ns" --wait ;;
  down) kind delete cluster --name "$CLUSTER" ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
