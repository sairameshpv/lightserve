terraform {
  required_version = ">= 1.5"

  required_providers {
    nebius = {
      source  = "nebius/nebius"
      version = ">= 0.6.8"
    }
  }

  # Production: use a remote backend so state is shared/locked across the team
  # instead of a local .tfstate file. Nebius object storage is S3-compatible,
  # so the standard "s3" backend works — fill in once a state bucket exists:
  #
  # backend "s3" {
  #   bucket                      = "<state-bucket-name>"
  #   key                         = "vllm/terraform.tfstate"
  #   endpoints                   = { s3 = "https://storage.<region>.nebius.cloud:443" }
  #   region                      = "<region>"
  #   skip_region_validation      = true
  #   skip_credentials_validation = true
  #   skip_requesting_account_id  = true
  #   skip_s3_checksum            = true
  # }
}