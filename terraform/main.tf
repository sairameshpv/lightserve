resource "nebius_compute_v1_disk" "boot_disk" {
  count = var.instance_count

  parent_id        = var.parent_id
  name             = "vllm-boot-disk-${count.index}"
  block_size_bytes = 4096
  size_bytes       = 1024 * 1024 * 1024 * var.boot_disk_size_gb
  type             = "NETWORK_SSD"

  # Boots from a pre-baked snapshot (Docker + NVIDIA Container Toolkit + Nsight
  # + vllm/vllm-openai:latest already installed/pulled) instead of the raw
  # ubuntu24.04-cuda12 image, so instances skip ~5-10min of idle-billed GPU
  # time on setup. Built via a cheap cpu-e2 instance, not the GPU itself —
  # see nebius_setup_commands.txt for the build process. Custom "image"
  # resources are blocked by a 0 quota on this tenant (support ticket needed
  # to raise it); disk snapshots hit no such limit, so this is the workaround.
  source_snapshot_id = var.golden_snapshot_id != "" ? var.golden_snapshot_id : null
  source_image_family = var.golden_snapshot_id == "" ? { image_family = "ubuntu24.04-cuda12" } : null
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
    hf_token          = var.hf_token
    model_name        = var.model_name
    bootstrap_tooling = var.golden_snapshot_id == ""
    enable_profiling  = var.enable_profiling
  })
}

# When enable_profiling is set, run_profiling.sh (baked into cloud-init above)
# captures nvidia-smi dmon + nsys + torch.profiler traces during boot and
# leaves them under /tmp/traces and /tmp/vllm-profiles. This pulls them
# down locally once cloud-init (and therefore the capture) has finished.
# No destroy-time provisioner, so `terraform destroy` is unaffected.
resource "null_resource" "pull_profiling_traces" {
  count = var.enable_profiling ? var.instance_count : 0

  triggers = {
    instance_id = nebius_compute_v1_instance.vllm[count.index].id
  }

  provisioner "local-exec" {
    command = <<-EOT
      set -euo pipefail
      HOST=$(echo '${nebius_compute_v1_instance.vllm[count.index].status.network_interfaces[0].public_ip_address.address}' | cut -d/ -f1)
      SSH="ssh -o StrictHostKeyChecking=accept-new -i ~/.ssh/nebius_key ubuntu@$HOST"
      echo "Waiting for cloud-init (profiling capture) to finish on $HOST..."
      $SSH "timeout 1800 sudo cloud-init status --wait"
      mkdir -p "${path.module}/../benchmarks/profiling/traces/torch_profiler"
      scp -i ~/.ssh/nebius_key "ubuntu@$HOST:/tmp/traces/vllm_session.nsys-rep" "${path.module}/../benchmarks/profiling/traces/" || true
      scp -i ~/.ssh/nebius_key "ubuntu@$HOST:/tmp/traces/dmon.log" "${path.module}/../benchmarks/profiling/traces/" || true
      scp -i ~/.ssh/nebius_key "ubuntu@$HOST:/tmp/traces/probe_timestamps.json" "${path.module}/../benchmarks/profiling/traces/session_a_probe_timestamps.json" || true
      scp -i ~/.ssh/nebius_key "ubuntu@$HOST:/tmp/traces/run_profiling.log" "${path.module}/../benchmarks/profiling/traces/" || true
      scp -i ~/.ssh/nebius_key "ubuntu@$HOST:/tmp/traces/session_a_*.csv" "${path.module}/../benchmarks/profiling/traces/" || true
      scp -i ~/.ssh/nebius_key "ubuntu@$HOST:/tmp/vllm-profiles/*.gz" "${path.module}/../benchmarks/profiling/traces/torch_profiler/" || true
      echo "Pulled available traces into benchmarks/profiling/traces/ (see run_profiling.log there for capture-time errors)."
    EOT
  }
}
