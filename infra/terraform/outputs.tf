output "cluster_id" {
  value = nebius_mk8s_v1_cluster.this.id
}

output "kubeconfig_command" {
  description = "Run this to use kubectl against the cluster."
  value       = "nebius mk8s cluster get-credentials --id ${nebius_mk8s_v1_cluster.this.id} --external"
}

output "registry_url" {
  description = "Push images here: docker push <registry_url>/lightserve:<tag>"
  value       = local.registry_url
}
