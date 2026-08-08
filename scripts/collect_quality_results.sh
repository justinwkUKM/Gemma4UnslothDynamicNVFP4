#!/usr/bin/env bash
set -Eeuo pipefail

# Copy quality artifacts without stopping or terminating the Pod.
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 user@host ssh_port local_quality_dir [identity_file]" >&2
  exit 2
fi
ssh_target="$1"
ssh_port="$2"
destination="$3"
identity_file="${4:-${SSH_IDENTITY_FILE:-}}"
remote_root="${GEMMA_QUALITY_REMOTE_ROOT:-/workspace/gemma4-quality}"

mkdir -p "$destination"/{environment,results,summary}
scp_args=(-P "$ssh_port" -o ConnectTimeout=15)
if [[ -n "$identity_file" ]]; then
  scp_args+=(-i "$identity_file")
fi
for directory in environment results summary; do
  scp "${scp_args[@]}" -r "$ssh_target:$remote_root/$directory/." "$destination/$directory/"
done

test -s "$destination/summary/quality-report.md" || {
  echo "quality report is absent or empty" >&2
  exit 1
}
find "$destination" -type f -print | sort
