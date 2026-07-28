output "instance_ids" {
  value = [for i in nebius_compute_v1_instance.vllm : i.id]
}

output "instance_names" {
  value = [for i in nebius_compute_v1_instance.vllm : i.name]
}

# The exact attribute path for reading back the assigned public IP from
# `status` wasn't confirmed against the provider schema — after the first
# `terraform apply`, run `terraform state show 'nebius_compute_v1_instance.vllm[0]'`
# to find the right path and wire it up here, or read the IP from the console/CLI.
