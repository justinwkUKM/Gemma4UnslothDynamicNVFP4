#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 3 || "$1" != "--authorized" ]]; then
  echo "usage: $0 --authorized POD_ID COLLECTED_CAMPAIGN_ROOT [ROOT ...]" >&2
  exit 2
fi

pod_id="$2"
shift 2
[[ "$pod_id" =~ ^[A-Za-z0-9]+$ ]] || {
  echo "invalid Pod ID" >&2
  exit 2
}
: "${RUNPOD_API_KEY:?export RUNPOD_API_KEY before using authenticated shutdown}"
command -v runpodctl >/dev/null || {
  echo "runpodctl is required" >&2
  exit 2
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 "$script_dir/verify_campaign.py" "$@"

pod_json="$(runpodctl pod get "$pod_id" -o json)"
actual_id="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("id", ""))' <<<"$pod_json")"
[[ "$actual_id" == "$pod_id" ]] || {
  echo "RunPod returned a different Pod identity" >&2
  exit 1
}

runpodctl pod stop "$pod_id" -o json >/dev/null
status_json="$(runpodctl pod get "$pod_id" -o json)"
python3 -c '
import json, sys
pod = json.load(sys.stdin)
if pod.get("desiredStatus") != "EXITED" or pod.get("runtimeStatus") not in {"stopped", None}:
    raise SystemExit("Pod did not reach EXITED/stopped state")
print(json.dumps({"id": pod.get("id"), "desiredStatus": pod.get("desiredStatus"), "runtimeStatus": pod.get("runtimeStatus"), "volumePreserved": True}, indent=2))
' <<<"$status_json"
