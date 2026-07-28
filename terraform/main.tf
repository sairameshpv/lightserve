resource "nebius_compute_v1_disk" "boot_disk" {
  count = var.instance_count

  parent_id           = var.parent_id
  name                = "vllm-boot-disk-${count.index}"
  block_size_bytes    = 4096
  size_bytes          = 1024 * 1024 * 1024 * var.boot_disk_size_gb
  type                = "NETWORK_SSD"
  source_image_family = { image_family = "ubuntu24.04-cuda12" }
}

# Only created when scaling to multiple GPU nodes that need InfiniBand interconnect.
resource "nebius_compute_v1_gpu_cluster" "gpu_cluster" {
  count = var.fabric != "" ? 1 : 0

  parent_id         = var.parent_id
  name              = "vllm-gpu-cluster"
  infiniband_fabric = var.fabric
}

resource "nebius_compute_v1_instance" "vllm" {
  count = var.instance_count

  parent_id = var.parent_id
  name      = "vllm-node-${count.index}"

  network_interfaces = [
    {
      name              = "eth0"
      subnet_id         = var.subnet_id
      ip_address        = {}
      public_ip_address = var.public_ip ? {} : null
    }
  ]

  resources = {
    platform = var.platform
    preset   = var.preset
  }

  boot_disk = {
    attach_mode   = "READ_WRITE"
    existing_disk = nebius_compute_v1_disk.boot_disk[count.index]
  }

  gpu_cluster = var.fabric != "" ? { id = nebius_compute_v1_gpu_cluster.gpu_cluster[0].id } : {}

  recovery_policy = "RECOVER"

  cloud_init_user_data = templatefile("${path.module}/cloud-init.tftpl", {
    hf_token   = var.hf_token
    model_name = var.model_name
  })
}
