variable "nebius_token" {
  type        = string
  sensitive   = true
  description = "Short-lived access token: export TF_VAR_nebius_token=$(nebius iam get-access-token). Never in a file."
}

variable "parent_id" {
  type        = string
  description = "Nebius project ID (same as terraform/terraform.tfvars)."
}

variable "subnet_id" {
  type        = string
  description = "Existing subnet for the control plane and the nodes (`nebius vpc subnet list --parent-id <parent_id>`)."
}

variable "name" {
  type        = string
  description = "Cluster name; also the prefix of the node groups."
  default     = "lightserve-mlops"
}

variable "k8s_version" {
  type        = string
  description = "Kubernetes <major>.<minor>. 1.32 supports gpu-l40s-a with the cuda12.8 driver preset (nebius mk8s node-group get-compatibility-matrix)."
  default     = "1.32"
}

variable "gpu_platform" {
  type    = string
  default = "gpu-l40s-a"
}

variable "gpu_preset" {
  type    = string
  default = "1gpu-24vcpu-96gb"
}

variable "gpu_drivers_preset" {
  type        = string
  description = "NVIDIA driver baked into the node image (the GPU operator then doesn't install one: ~10 min saved)."
  default     = "cuda12.8"
}

variable "gpu_node_count" {
  type        = number
  description = "0 = a cluster without the GPU node (cheaper rehearsal of everything else)."
  default     = 1
}

variable "preemptible" {
  type        = bool
  description = "Preemptible (spot) GPU node: cheaper, can be reclaimed; on-demand L40S quota can hit LIMIT_REACHED (see terraform/variables.tf)."
  default     = false
}

variable "grafana_admin_password" {
  type        = string
  sensitive   = true
  description = "Grafana admin password (TF_VAR_grafana_admin_password)."
}

variable "lightserve_image_repository" {
  type        = string
  description = "Where CI pushes lightserve's image (.github/workflows/image.yml)."
  default     = "ghcr.io/sairameshpv/lightserve"
}

variable "lightserve_image_tag" {
  type        = string
  description = "gpu-<git sha> or gpu-latest (CI pushes both for the gpu target)."
  default     = "gpu-latest"
}

variable "lightserve_n_layers" {
  type        = number
  description = "lightserve runs Llama-3-8B's shape with random weights (it has no weight loader): 4 layers = toy, fast start; 32 = the full 8B shape."
  default     = 4
}
