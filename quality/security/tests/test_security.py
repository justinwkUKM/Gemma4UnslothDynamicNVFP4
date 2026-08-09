import json
import tempfile
import unittest
from pathlib import Path

from quality.security.context import build_investigations
from quality.security.contract import ContractError, validate_analysis
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


if __name__ == "__main__":
    unittest.main()
