# Quick-start auth: a short-lived personal access token, passed via the
# TF_VAR_nebius_token env var (never write it into a file). Good for initial
# testing; swap to a service account for anything long-running/production
# (see the commented block below) since this token expires (~1hr).
provider "nebius" {
  token = var.nebius_token
}

# Production alternative — service account, longer-lived, no manual token refresh:
#   export AUTHKEY_PRIVATE_PATH=/path/to/service-account-private-key.pem
#   export AUTHKEY_PUBLIC_ID=<public-key-id>
#   export SA_ID=<service-account-id>
# provider "nebius" {
#   service_account = {
#     private_key_file_env = "AUTHKEY_PRIVATE_PATH"
#     public_key_id_env    = "AUTHKEY_PUBLIC_ID"
#     account_id_env       = "SA_ID"
#   }
# }
# Reference: https://docs.nebius.com/terraform-provider/install