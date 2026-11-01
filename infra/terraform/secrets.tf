# Secrets come from TF_VAR_* environment variables into Kubernetes secrets, never into git.
# (They do land in the local terraform.tfstate, which is gitignored, as in terraform/.)

resource "kubernetes_namespace" "ns" {
  for_each = toset(["lightserve", "monitoring", "mlops", "chaos-mesh", "gpu-operator"])
  metadata { name = each.key }
  depends_on = [nebius_mk8s_v1_node_group.cpu]
}

resource "kubernetes_secret" "hf_token" {
  metadata {
    name      = "hf-token"
    namespace = kubernetes_namespace.ns["lightserve"].metadata[0].name
  }
  data = { token = var.hf_token }
}

resource "kubernetes_secret" "grafana_admin" {
  metadata {
    name      = "grafana-admin"
    namespace = kubernetes_namespace.ns["monitoring"].metadata[0].name
  }
  data = { admin-user = "admin", admin-password = var.grafana_admin_password }
}
