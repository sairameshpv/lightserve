# Auth is via env vars, not values in this file — never commit credentials.
# Set before running terraform:
#   export AUTHKEY_PRIVATE_PATH=/path/to/service-account-private-key.pem
#   export AUTHKEY_PUBLIC_ID=<public-key-id>
#   export SA_ID=<service-account-id>
# Reference: https://docs.nebius.com/terraform-provider/install
provider "nebius" {
  service_account = {
    private_key_file_env = "AUTHKEY_PRIVATE_PATH"
    public_key_id_env    = "AUTHKEY_PUBLIC_ID"
    account_id_env       = "SA_ID"
  }
}