terraform {
  required_version = ">= 1.5"

  required_providers {
    nebius = {
      source  = "nebius/nebius"
      version = ">= 0.6.67"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.38"
    }
  }
  # State stays local (gitignored), as in terraform/: one person, one cluster at a time.
}
