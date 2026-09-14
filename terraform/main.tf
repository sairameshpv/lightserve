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

  # Nebius's schema: include this block to get a preemptible VM, omit
  # (null) for a regular one -- not a plain bool despite var.preemptible
  # being one. on_preemption="STOP" is the only supported value (Nebius
  # stops rather than deletes/restarts the VM when it reclaims capacity).
  preemptible = var.preemptible ? { on_preemption = "STOP" } : null

  boot_disk = {
    attach_mode   = "READ_WRITE"
    existing_disk = nebius_compute_v1_disk.boot_disk[count.index]
  }

  gpu_cluster = var.fabric != "" ? { id = nebius_compute_v1_gpu_cluster.gpu_cluster[0].id } : {}

  # Nebius requires "FAIL" for preemptible instances -- "RECOVER" is
  # rejected outright (InvalidArgument), unrelated to the HF-token work,
  # found the hard way when this apply hit it head-on.
  recovery_policy = var.preemptible ? "FAIL" : "RECOVER"

  # hf_token/model_name deliberately not passed here -- see push_hf_token
  # below, which injects them post-boot over SSH instead of baking them
  # into cloud_init_user_data (a permanently-stored, plaintext-readable
  # instance attribute).
  cloud_init_user_data = templatefile("${path.module}/cloud-init.tftpl", {
    bootstrap_tooling = var.golden_snapshot_id == ""
    enable_profiling  = var.enable_profiling
  })
}

# Pushes the HF token over SSH after boot, then launches the container --
# see cloud-init.tftpl's comment for why: keeps the token out of
# cloud_init_user_data (a permanently-stored, plaintext-readable instance
# attribute) and out of terraform.tfstate. environment (not string
# interpolation) so the token value never appears in the rendered command
# itself.
resource "null_resource" "push_hf_token" {
  count = var.instance_count

  triggers = {
    instance_id = nebius_compute_v1_instance.vllm[count.index].id
  }

  provisioner "local-exec" {
    environment = {
      HF_TOKEN   = var.hf_token
      MODEL_NAME = var.model_name
    }

    command = <<-EOT
      set -euo pipefail
      HOST=$(echo '${nebius_compute_v1_instance.vllm[count.index].status.network_interfaces[0].public_ip_address.address}' | cut -d/ -f1)
      SSH="ssh -o StrictHostKeyChecking=accept-new -i ~/.ssh/nebius_key ubuntu@$HOST"
      # Plain retry loop, not `timeout` -- that's GNU coreutils, not on
      # macOS by default (the remote `timeout` calls below are fine, those
      # run on the Ubuntu box, which has it).
      echo "Waiting for SSH + cloud-init on $HOST..."
      for i in $(seq 1 60); do $SSH true 2>/dev/null && break; sleep 5; done
      $SSH "timeout 1800 sudo cloud-init status --wait"

      echo "Pushing HF token to a root-only file (never touches cloud-init or terraform state)..."
      printf 'HUGGING_FACE_HUB_TOKEN=%s\n' "$HF_TOKEN" | $SSH "sudo install -o root -g root -m 600 /dev/stdin /root/hf_token.env"

      # The golden snapshot has a leftover "vllm-server" container baked
      # into its disk image from when it was built -- harmless normally
      # (only surfaces on a genuinely fresh boot from the snapshot, not a
      # stop/start of an existing instance), but collides with the name
      # below if not cleared first.
      $SSH "sudo docker rm -f vllm-server" || true

      %{ if var.enable_profiling ~}
      echo "Running profiling capture (blocking, ~15-20min)..."
      $SSH "sudo MODEL_NAME='$MODEL_NAME' /root/run_profiling.sh > /tmp/traces/run_profiling.log 2>&1"
      %{ else ~}
      echo "Launching vllm-server..."
      $SSH "sudo docker run -d --name vllm-server --restart unless-stopped --gpus all \
        -v /root/.cache/huggingface:/root/.cache/huggingface \
        --env-file /root/hf_token.env \
        -p 8000:8000 --ipc=host \
        vllm/vllm-openai:latest --model '$MODEL_NAME'"
      %{ endif ~}
    EOT
  }
}

# When enable_profiling is set, run_profiling.sh (baked into cloud-init above)
# captures nvidia-smi dmon + nsys + torch.profiler traces during boot and
# leaves them under /tmp/traces and /tmp/vllm-profiles. This pulls them
# down locally once cloud-init (and therefore the capture) has finished.
# No destroy-time provisioner, so `terraform destroy` is unaffected.
resource "null_resource" "pull_profiling_traces" {
  count = var.enable_profiling ? var.instance_count : 0

  # The capture itself now runs inside push_hf_token (it needs the token),
  # not inside cloud-init's runcmd -- so this must wait for that to finish,
  # not just for cloud-init, or it'd race and pull an empty traces dir.
  depends_on = [null_resource.push_hf_token]

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
