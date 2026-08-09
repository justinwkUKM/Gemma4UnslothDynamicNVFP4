import json
import tempfile
import unittest
from pathlib import Path

from campaigns.common import CampaignManifest, ManifestError, atomic_write_jsonl, canonical_sha256, read_jsonl


class CampaignCommonTests(unittest.TestCase):
    def make_manifest(self, root: Path) -> CampaignManifest:
        return CampaignManifest.create(
            root / "campaign.json",
            campaign_id="test-campaign",
            campaign_type="test",
            dataset_versions={"fixture": "1"},
            model_versions={"model": "revision"},
            backend="fixture",
            context_limit=4096,
            seed=0,
            prompt_hash=canonical_sha256(["p1"]),
            environment_hash=canonical_sha256({"gpu": "fixture"}),
            gpu_type="fixture",
            started_at_utc="2026-01-01T00:00:00+00:00",
            deadline_utc="2026-01-01T01:00:00+00:00",
            hourly_rate_usd=0.69,
            config_hash=canonical_sha256({"seed": 0}),
        )

    def test_complete_requires_all_acceptance_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.make_manifest(Path(directory))
            with self.assertRaises(ManifestError):
                manifest.finish("complete", requirements={"all_repetitions": False})
            manifest.finish("partial", requirements={"all_repetitions": False})
            loaded = json.loads(manifest.path.read_text())
            self.assertEqual(loaded["status"], "partial")

    def test_jsonl_replace_is_ordered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            atomic_write_jsonl(path, ({"id": "a"}, {"id": "b"}))
            self.assertEqual([row["id"] for row in read_jsonl(path)], ["a", "b"])

    def test_resume_preserves_campaign_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.make_manifest(root)
            first.finish("partial", requirements={"done": False})
            resumed = self.make_manifest(root)
            self.assertEqual(resumed.data["status"], "running")
            self.assertEqual(resumed.data["resume_count"], 1)


if __name__ == "__main__":
    unittest.main()
