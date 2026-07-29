variable "parent_id" {
  type        = string
  description = "Nebius project (folder) ID resources are created in."
}

variable "subnet_id" {
  type        = string
  description = "ID of an existing subnet. Look up via `nebius vpc subnet list --parent-id <parent_id>` or the console — this module does not create the network."
}

variable "instance_count" {
  type        = number
  description = "Number of vLLM GPU instances. Start at 1, raise to 2+ to scale out."
  default     = 1
}

variable "platform" {
  type        = string
  description = "GPU platform. Verified via `nebius capacity resource-advice list` against project-e00k2ff2pr005cst6j72s0 (eu-north1): gpu-l40s-a has usable on-demand quota (limit 32 GPUs tenant-wide); gpu-h100-sxm quota is unrequested (USAGE_STATE_NOT_USED) there."
  default     = "gpu-l40s-a"
}

variable "preset" {
  type        = string
  description = "Resource preset for the chosen platform (vCPU/GPU/RAM bundle). Verified available for gpu-l40s-a in eu-north1."
  default     = "1gpu-24vcpu-96gb"
}

variable "golden_snapshot_id" {
  type        = string
  description = "Disk snapshot ID to boot from (pre-baked with Docker/NVIDIA Container Toolkit/Nsight/vLLM image). Leave empty to fall back to the raw ubuntu24.04-cuda12 image (slower first boot, full cloud-init install)."
  default     = ""
}

variable "boot_disk_size_gb" {
  type        = number
  description = "Boot disk size in GB. Boots from the ubuntu24.04-cuda12 image family (CUDA preinstalled), so leave headroom for the Docker image + model cache if not using a separate data disk."
  default     = 200
}

variable "fabric" {
  type        = string
  description = "InfiniBand fabric name. Only set this when scaling to multiple GPU instances that need low-latency interconnect; leave empty for a single node."
  default     = ""
}

variable "public_ip" {
  type        = bool
  description = "Attach a public IP. In production, prefer false + access via bastion/VPN; true is simplest to get started."
  default     = true
}

variable "model_name" {
  type        = string
  description = "Hugging Face model id to serve."
  default     = "meta-llama/Meta-Llama-3-8B-Instruct"
}

variable "nebius_token" {
  type        = string
  description = "Nebius personal access token for provider auth (quick-start path). Pass via TF_VAR_nebius_token env var — never hardcode or commit. Short-lived (~1hr); re-export before each session."
  sensitive   = true
}

variable "hf_token" {
  type        = string
  description = "Hugging Face access token (must have accepted the Llama-3 license). Pass via TF_VAR_hf_token env var or a secrets manager — never hardcode or commit."
  sensitive   = true
}