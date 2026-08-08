#!/usr/bin/env bash
set -Eeuo pipefail

# Copy durable artifacts without stopping or terminating the Pod. The caller
# must provide the SSH destination and a timestamped local output directory.
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 user@host ssh_port local_output_dir [identity_file]" >&2
  exit 2
fi
ssh_target="$1"
ssh_port="$2"
destination="$3"
identity_file="${4:-${SSH_IDENTITY_FILE:-$HOME/.ssh/id_ed25519}}"
remote_root="${GEMMA_REMOTE_ROOT:-/workspace/gemma4-benchmark}"

mkdir -p "$destination"
scp_args=(-P "$ssh_port" -i "$identity_file" -o ConnectTimeout=15)
scp "${scp_args[@]}" -r "$ssh_target:$remote_root/config" "$destination/"
scp "${scp_args[@]}" -r "$ssh_target:$remote_root/environment" "$destination/"
scp "${scp_args[@]}" -r "$ssh_target:$remote_root/logs" "$destination/"
scp "${scp_args[@]}" -r "$ssh_target:$remote_root/results" "$destination/"
scp "${scp_args[@]}" -r "$ssh_target:$remote_root/summary" "$destination/"

test -s "$destination/results/normalized.json" || {
  echo "warning: normalized.json is absent or empty; the run may still be active" >&2
}
find "$destination" -type f -print | sort
