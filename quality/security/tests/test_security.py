import json
import gzip
import io
import os
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from quality.security.context import build_investigations
from quality.security.contract import ContractError, validate_analysis
from quality.security.adapters import adapt_lanl, adapt_optc, adapt_otrf
from quality.security.datasets import build_status, prepare_lanl
from quality.security.evaluator import evaluate
from quality.security.parser import CanonicalTelemetryParser, anonymize_public_events
from quality.security.replay import ReplayEngine
from quality.security.runner import SECURITY_MODELS, assert_inference_safe
from quality.security.prepare import remap_truth_ids


def event(event_id="E1", timestamp="2026-01-01T00:00:00+00:00", action="PowerShell encoded command"):
    return {
        "schema_version": 1,
        "event_id": event_id,
        "timestamp": timestamp,
        "dataset": "fixture",
        "source_type": "endpoint",
        "entity_id": "HOST-1",
        "action": action,
        "outcome": "success",
        "attributes": {},
    }


def analysis(evidence="E1", status="suspicious"):
    return {
        "status": status,
        "risk_score": 70,
        "confidence": 0.8,
        "observations": [{"summary": "encoded command", "evidence_ids": [evidence]}],
        "hypotheses": [{"description": "possible execution", "confidence": 0.6, "evidence_ids": [evidence]}],
        "attack_techniques": [{"technique_id": "T1059", "evidence_ids": [evidence]}],
        "related_entities": ["HOST-1"],
        "recommendations": [{"action": "isolate", "evidence_ids": [evidence]}],
        "missing_information": [],
        "predicted_next_actions": [{"action": "credential access", "confidence": 0.5, "evidence_ids": [evidence]}],
    }


class SecurityHarnessTests(unittest.TestCase):
    def test_parser_strips_labels(self):
        parser = CanonicalTelemetryParser(dataset="otrf")
        parsed = parser.parse({
            "id": "1", "timestamp": "2026-01-01T00:00:00Z", "host": "h",
            "action": "login", "malicious": True, "label": "attack",
        }, 1)
        self.assertNotIn("malicious", parsed["attributes"])
        self.assertNotIn("label", parsed["attributes"])

    def test_public_anonymization_preserves_intervals(self):
        values = [event("E1"), event("E2", "2026-01-01T00:00:30+00:00")]
        anonymized = anonymize_public_events(values, salt="fixture")
        windows = list(ReplayEngine(anonymized).windows(30))
        self.assertEqual(len(windows), 2)
        self.assertNotEqual(anonymized[0]["entity_id"], "HOST-1")
        assert_inference_safe(anonymized, track="public")

    def test_public_anonymization_scrubs_identifiers_inside_messages(self):
        value = event()
        value["attributes"] = {
            "SourceAddress": "10.0.0.4",
            "TargetUserName": "ALICE",
            "tags": ["mordorDataset"],
            "CommandLine": "powershell.exe -encodedCommand SQBFAFgA",
            "Message": "ALICE connected from 10.0.0.4 using powershell.exe",
        }
        anonymized = anonymize_public_events([value], salt="fixture")[0]
        rendered = json.dumps(anonymized)
        self.assertNotIn("ALICE", rendered)
        self.assertNotIn("10.0.0.4", rendered)
        self.assertNotIn("mordorDataset", rendered)
        self.assertIn("powershell.exe -encodedCommand", rendered)
        self.assertEqual(anonymized["dataset"], "public-telemetry")

    def test_triggered_context_omits_benign_noise(self):
        values = [event("E1", action="routine heartbeat"), event("E2", action="PowerShell encoded command")]
        jobs = build_investigations(values, mode="triggered", window_seconds=30)
        self.assertEqual(jobs[0]["event_ids"], ["E2"])

    def test_security_matrix_contains_only_gemma_4_models(self):
        self.assertEqual(set(SECURITY_MODELS), {
            "gemma-4-E4B-it-NVFP4",
            "gemma-4-12b-it-NVFP4",
            "gemma-4-26B-A4B-it-NVFP4",
        })
        self.assertTrue(all("gemma-4" in checkpoint.lower() for checkpoint in SECURITY_MODELS.values()))

    def test_contract_rejects_invented_evidence(self):
        verdict = validate_analysis(analysis(), {"E1"})
        self.assertTrue(verdict["valid"])
        invented = validate_analysis(analysis("E999"), {"E1"})
        self.assertFalse(invented["valid"])

    def test_ground_truth_ids_use_same_anonymization(self):
        values = anonymize_public_events([event("E1")], salt="fixture")
        truth = remap_truth_ids({"attack_event_ids": ["E1"]}, "fixture")
        self.assertEqual(truth["attack_event_ids"], [values[0]["event_id"]])

    def test_evaluator_keeps_scorecards_separate(self):
        records = [{
            "scenario_id": "s1", "status": "success", "analysis_timestamp": "2026-01-01T00:00:10+00:00",
            "analysis": analysis(), "context_event_ids": ["E1"], "latency_seconds": 1.0,
            "request_cost_usd": 0.01, "output_text": json.dumps(analysis()),
        }]
        truth = {"scenarios": [{
            "scenario_id": "s1", "malicious": True, "attack_start": "2026-01-01T00:00:00+00:00",
            "impact_time": "2026-01-01T00:01:00+00:00", "start_time": "2026-01-01T00:00:00+00:00",
            "end_time": "2026-01-01T00:01:00+00:00", "attack_event_ids": ["E1"],
            "next_actions": ["credential access"],
        }]}
        score = evaluate(records, truth)
        self.assertEqual(score["security_intelligence"]["detection_recall"], 1.0)
        self.assertIn("operational", score)
        self.assertNotIn("combined_score", score)

    def test_real_dataset_adapters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            otrf = root / "otrf.zip"
            with zipfile.ZipFile(otrf, "w") as archive:
                archive.writestr("events.json", json.dumps({
                    "@timestamp": "2020-01-01T00:00:00Z",
                    "Hostname": "HOST-1", "Channel": "Security",
                    "EventID": 4688, "Category": "Process Creation",
                }) + "\n")
            otrf_event = next(adapt_otrf(otrf))
            self.assertEqual(otrf_event["entity_id"], "HOST-1")
            self.assertEqual(otrf_event["action"], "Process Creation")

            lanl = root / "auth.txt.gz"
            with gzip.open(lanl, "wt", encoding="utf-8") as handle:
                handle.write("1,U1@DOM1,U2@DOM1,C1,C2,Kerberos,Network,LogOn,Success\n")
            lanl_event = next(adapt_lanl(lanl, "auth"))
            self.assertEqual(lanl_event["source_type"], "auth")
            self.assertEqual(lanl_event["outcome"], "Success")

            optc = root / "optc.jsonl"
            optc.write_text(json.dumps({
                "timestamp": 1539120748904,
                "id": "e1", "hostname": "H1", "object": "PROCESS", "action": "CREATE",
            }) + "\n", encoding="utf-8")
            optc_event = next(adapt_optc(optc))
            self.assertEqual(optc_event["timestamp"], "2018-10-09T21:32:28.904000+00:00")
            self.assertEqual(optc_event["source_type"], "PROCESS")

            optc_tar = root / "2019-09-16.tar"
            payload = optc.read_bytes()
            with tarfile.open(optc_tar, "w") as archive:
                info = tarfile.TarInfo("ecar/day.json")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            self.assertEqual(next(adapt_optc(optc_tar))["event_id"], "e1")

    def test_dataset_status_requires_real_files(self):
        with tempfile.TemporaryDirectory() as directory:
            status = build_status(Path(directory))
        self.assertEqual(status["datasets"]["otrf"]["acquisition"], "missing")
        self.assertEqual(status["datasets"]["lanl"]["present_files"], [])
        self.assertFalse(status["datasets"]["cyber_range"]["scored_inference_ready"])

    def test_lanl_merge_keeps_redteam_truth_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = {
                "auth": (
                    "1,U1@DOM1,U2@DOM1,C1,C2,Kerberos,Network,LogOn,Success\n"
                    "1,U1@DOM1,U2@DOM1,C1,C2,Kerberos,Network,LogOn,Success\n"
                ),
                "proc": "1,U1@DOM1,C1,P1,Start\n",
                "flows": "1,1,C1,123,C2,443,6,2,90\n",
                "dns": "1,C1,C2\n",
                "redteam": "1,U1@DOM1,C1,C2\n",
            }
            for name, content in values.items():
                with gzip.open(root / f"{name}.txt.gz", "wt", encoding="utf-8") as handle:
                    handle.write(content)
            args = SimpleNamespace(
                input=root,
                output=root / "canonical.jsonl",
                ground_truth_output=root / "sealed-truth.json",
                salt_env="SECURITY_BENCHMARK_SALT",
                limit=None,
            )
            with patch.dict(os.environ, {"SECURITY_BENCHMARK_SALT": "fixture-salt-long-enough"}):
                prepare_lanl(args)
            truth = json.loads(args.ground_truth_output.read_text(encoding="utf-8"))
            inference = args.output.read_text(encoding="utf-8")
            self.assertEqual(truth["matched_auth_event_count"], 2)
            self.assertEqual(truth["matched_redteam_key_count"], 1)
            self.assertEqual(truth["unmatched_redteam_event_count"], 0)
            self.assertNotIn("known_redteam", inference)
            self.assertNotIn("U1@DOM1", inference)
            self.assertFalse(truth["scored_inference_allowed"])


if __name__ == "__main__":
    unittest.main()
