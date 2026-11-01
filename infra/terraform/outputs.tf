output "cluster_id" {
  value = nebius_mk8s_v1_cluster.this.id
}

output "kubeconfig_command" {
  description = "Run this to use kubectl against the cluster."
  value       = "nebius mk8s cluster get-credentials --id ${nebius_mk8s_v1_cluster.this.id} --external"
}
