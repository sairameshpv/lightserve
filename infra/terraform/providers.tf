provider "nebius" {
  token = var.nebius_token # short-lived (~1 h): fine for a < 1 h session; see terraform/provider.tf for a service account
}

# Kubernetes and Helm talk to the new cluster's public endpoint; the nebius CLI hands out a token
# (the same command `nebius mk8s cluster get-credentials` writes into a kubeconfig).
locals {
  k8s_host = nebius_mk8s_v1_cluster.this.status.control_plane.endpoints.public_endpoint
  k8s_ca   = nebius_mk8s_v1_cluster.this.status.control_plane.auth.cluster_ca_certificate
  k8s_exec = {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "nebius"
    args        = ["mk8s", "v1", "cluster", "get-token", "--format", "json"]
  }
}

provider "kubernetes" {
  host                   = local.k8s_host
  cluster_ca_certificate = local.k8s_ca
  exec {
    api_version = local.k8s_exec.api_version
    command     = local.k8s_exec.command
    args        = local.k8s_exec.args
  }
}

provider "helm" {
  kubernetes = {
    host                   = local.k8s_host
    cluster_ca_certificate = local.k8s_ca
    exec                   = local.k8s_exec
  }
}
