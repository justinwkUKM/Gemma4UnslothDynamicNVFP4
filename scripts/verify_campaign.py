#!/usr/bin/env python3
"""Verify a collected campaign's manifest and artifact hashes before shutdown."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from campaigns.common import CampaignManifest, sha256_file  # noqa: E402


def verify(root: Path) -> list[str]:
    errors = []
    manifest_path = root / "campaign.json"
    try:
        manifest = CampaignManifest.load(manifest_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [f"invalid campaign manifest: {exc}"]
    if manifest.data["status"] not in {"complete", "partial", "failed"}:
        errors.append(f"campaign is still {manifest.data['status']}")
    hashes = manifest.data.get("artifacts", {})
    if not hashes:
        errors.append("manifest has no collected artifact hashes")
    for relative, expected in hashes.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing artifact: {relative}")
        elif sha256_file(path) != expected:
            errors.append(f"hash mismatch: {relative}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_roots", type=Path, nargs="+")
    args = parser.parse_args(argv)
    all_errors = []
    for root in args.campaign_roots:
        all_errors.extend(f"{root}: {error}" for error in verify(root))
    if all_errors:
        print("\n".join(all_errors), file=sys.stderr)
        return 1
    print(json.dumps({"verified": [str(root) for root in args.campaign_roots]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
