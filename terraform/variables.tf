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
  description = "GPU platform, e.g. \"gpu-h100-sxm\". Verify availability/quota for your project via the console before applying."
  default     = "gpu-h100-sxm"
}

variable "preset" {
  type        = string
  description = "Resource preset for the chosen platform (vCPU/GPU/RAM bundle). Confirm the exact preset name for your quota — this default is UNVERIFIED, check `nebius compute preset list` or the console before first apply."
  default     = "1gpu-16vcpu-200gb"
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

variable "hf_token" {
  type        = string
  description = "Hugging Face access token (must have accepted the Llama-3 license). Pass via TF_VAR_hf_token env var or a secrets manager — never hardcode or commit."
  sensitive   = true
}