# One terraform apply: a Nebius managed Kubernetes cluster with one L40S node, which runs everything, and
# (helm.tf) everything the platform runs. terraform destroy removes it all. lightserve's image comes
# from ghcr.io (CI pushes it there: .github/workflows/image.yml), which outlives the cluster.

resource "nebius_mk8s_v1_cluster" "this" {
  parent_id = var.parent_id
  name      = var.name
  control_plane = {
    version   = var.k8s_version
    subnet_id = var.subnet_id
    endpoints = {
      public_endpoint = {} # reachable from this laptop (kubectl, helm)
    }
    etcd_cluster_size = 1 # a short-lived test cluster; production: 3 (survives one control-plane failure)
  }
}

# No CPU node group: Nebius caps regular (non-GPU) vCPUs at 3 in eu-north1, the only region with
# L40S GPUs, and declined a raise (2026-11-03). GPU nodes count against the GPU quota instead, so the
# one L40S node (24 vCPUs, 96 GB) runs everything: lightserve and the whole platform.
resource "nebius_mk8s_v1_node_group" "gpu" {
  parent_id        = nebius_mk8s_v1_cluster.this.id
  name             = "${var.name}-gpu"
  fixed_node_count = var.gpu_node_count
  template = {
    resources    = { platform = var.gpu_platform, preset = var.gpu_preset }
    gpu_settings = { drivers_preset = var.gpu_drivers_preset }
    boot_disk    = { type = "NETWORK_SSD", size_gibibytes = 200 } # images + model cache
    network_interfaces = [{
      subnet_id         = var.subnet_id
      public_ip_address = {}
    }]
    metadata = { labels = { "lightserve/pool" = "gpu" } }
    # No taint: it's the only node, so every pod (Kubernetes's own included) must be able to land here.
    preemptible        = var.preemptible ? {} : null
    follows_spot_price = var.preemptible ? {} : null # Nebius requires it on preemptible nodes
  }
}
