output "instance_ids" {
  value = [for i in nebius_compute_v1_instance.vllm : i.id]
}

output "instance_names" {
  value = [for i in nebius_compute_v1_instance.vllm : i.name]
}

output "instance_public_ips" {
  value = [for i in nebius_compute_v1_instance.vllm : i.status.network_interfaces[0].public_ip_address.address]
}
