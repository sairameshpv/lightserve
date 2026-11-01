# Everything the platform runs, from infra/helm/releases.yaml (the same list the kind rehearsal
# installs from): same charts, versions and settings. Stage 1 (GPU operator, monitoring) first,
# because the other charts only create their alert rules and ServiceMonitors once monitoring's
# definitions exist.

locals {
  helm_dir = "${path.module}/../helm"
  releases = {
    for r in yamldecode(file("${local.helm_dir}/releases.yaml")) : r.name => r
    if !try(r.gpu, false) || var.gpu_node_count > 0
  }
  stage1 = { for k, r in local.releases : k => r if try(r.stage, 2) == 1 }
  stage2 = { for k, r in local.releases : k => r if try(r.stage, 2) != 1 }

  # what the real cluster adds on top of the shared values (the rehearsal adds its own)
  extra_values = {
    platform = {
      alerts = file("${path.module}/../../monitoring/alerts.yml")
      slo    = file("${path.module}/../../monitoring/slo.yml")
      dashboards = {
        lightserve = file("${path.module}/../../monitoring/grafana/provisioning/dashboards/lightserve.json")
        cost       = file("${path.module}/../../monitoring/grafana/provisioning/dashboards/cost.json")
      }
    }
    lightserve = {
      image   = { repository = var.lightserve_image_repository, tag = var.lightserve_image_tag }
      nLayers = var.lightserve_n_layers
    }
  }
}

resource "helm_release" "stage1" {
  for_each   = local.stage1
  name       = each.key
  namespace  = each.value.namespace
  repository = try(each.value.repo, null)
  chart      = try(each.value.repo, null) == null ? "${local.helm_dir}/${trimprefix(each.value.chart, "../")}" : each.value.chart
  version    = try(each.value.version, null)
  values     = [file("${local.helm_dir}/values/${each.value.values}"), yamlencode(try(local.extra_values[each.key], {}))]
  wait       = try(each.value.wait, true)
  timeout    = 900
  depends_on = [kubernetes_namespace.ns, kubernetes_secret.grafana_admin, nebius_mk8s_v1_node_group.gpu]
}

resource "helm_release" "stage2" {
  for_each   = local.stage2
  name       = each.key
  namespace  = each.value.namespace
  repository = try(each.value.repo, null)
  chart      = try(each.value.repo, null) == null ? "${local.helm_dir}/${trimprefix(each.value.chart, "../")}" : each.value.chart
  version    = try(each.value.version, null)
  values     = [file("${local.helm_dir}/values/${each.value.values}"), yamlencode(try(local.extra_values[each.key], {}))]
  wait       = try(each.value.wait, true)
  timeout    = 900
  depends_on = [helm_release.stage1]
}
