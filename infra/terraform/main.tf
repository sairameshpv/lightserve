# One terraform apply: a Nebius managed Kubernetes cluster (one CPU node + one L40S node), a
# container registry, and (helm.tf) everything the platform runs. terraform destroy removes it all.

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

resource "nebius_mk8s_v1_node_group" "cpu" {
  parent_id        = nebius_mk8s_v1_cluster.this.id
  name             = "${var.name}-cpu"
  fixed_node_count = 1
  template = {
    resources = { platform = var.cpu_platform, preset = var.cpu_preset }
    boot_disk = { type = "NETWORK_SSD", size_gibibytes = 96 }
    network_interfaces = [{
      subnet_id         = var.subnet_id
      public_ip_address = {} # to pull public images
    }]
    metadata = { labels = { "lightserve/pool" = "cpu" } }
  }
}

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
    # only pods that ask for a GPU (and tolerate this) land here
    taints             = [{ key = "nvidia.com/gpu", value = "present", effect = "NO_SCHEDULE" }]
    preemptible        = var.preemptible ? {} : null
    follows_spot_price = var.preemptible ? {} : null # Nebius requires it on preemptible nodes
  }
}

resource "nebius_registry_v1_registry" "this" {
  parent_id   = var.parent_id
  name        = var.name
  description = "lightserve images (pushed by CI)"
}

locals {
  # images are named <registry_fqdn>/<registry id without "registry-">/<image>:<tag> (Nebius docs)
  registry_url = "${nebius_registry_v1_registry.this.status.registry_fqdn}/${trimprefix(nebius_registry_v1_registry.this.id, "registry-")}"
}
