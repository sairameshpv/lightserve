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
  description = "Cluster name; also the prefix of the node groups and the registry."
  default     = "lightserve-mlops"
}

variable "k8s_version" {
  type        = string
  description = "Kubernetes <major>.<minor>. 1.32 supports gpu-l40s-a with the cuda12.8 driver preset (nebius mk8s node-group get-compatibility-matrix)."
  default     = "1.32"
}

variable "cpu_platform" {
  type    = string
  default = "cpu-d3"
}

variable "cpu_preset" {
  type        = string
  description = "One CPU node runs monitoring, MLflow, Airflow, Tempo, Loki, Chaos Mesh."
  default     = "8vcpu-32gb"
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

variable "hf_token" {
  type        = string
  sensitive   = true
  description = "Hugging Face token for the real Llama weights (TF_VAR_hf_token). Empty = toy model only."
  default     = ""
}

variable "grafana_admin_password" {
  type        = string
  sensitive   = true
  description = "Grafana admin password (TF_VAR_grafana_admin_password)."
}

variable "lightserve_image" {
  type        = string
  description = "lightserve container image (built and pushed by CI to the registry). Empty = the registry's lightserve:latest."
  default     = ""
}

variable "lightserve_real_model" {
  type        = bool
  description = "true = real Llama-3-8B weights (needs hf_token, ~16 GB download); false = the toy model (fast start)."
  default     = false
}
