import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluator


class EvaluatorTests(unittest.TestCase):
    def record(self, scorer, **scoring):
        return {
            "id": "test-001",
            "category": "arithmetic_reasoning",
            "prompt": "test",
            "acceptable_answers": ["Alpha Beta", "alternate"],
            "normalization": {
                "casefold": True,
                "strip_punctuation": True,
                "collapse_whitespace": True,
                "strip_articles": True,
            },
            "key_facts": [],
            "sources": [],
            "expected_behavior": "answer",
            "scoring": {"type": scorer, **scoring},
        }

    def test_case_whitespace_punctuation_and_articles_normalization(self):
        value = evaluator.normalize_text("  The  ALPHA,   beta! ", self.record("normalized_exact")["normalization"])
        self.assertEqual(value, "alpha beta")

    def test_acceptable_answer_variants(self):
        result = evaluator.score_exact(self.record("normalized_exact"), " Alternate! ")
        self.assertTrue(result["exact_match"])

    def test_numeric_answer_and_tolerance(self):
        record = self.record("numeric", value=4.2, tolerance=0.001)
        self.assertTrue(evaluator.score_numeric(record, "4.2005")["exact_match"])
        self.assertFalse(evaluator.score_numeric(record, "4.2 or 4.3")["exact_match"])

    def test_required_fact_detection_and_contradiction(self):
        record = self.record("required_facts")
        record["key_facts"] = [
            {"id": "planet", "variants": ["Jupiter"], "incorrect_variants": ["Saturn"]},
            {"id": "position", "variants": ["fifth", "5th"], "incorrect_variants": ["sixth"]},
        ]
        result = evaluator.score_required_facts(record, "Jupiter is fifth; it is not Saturn.")
        self.assertEqual(result["fact_recall"], 1.0)
        self.assertEqual(result["fact_precision"], 2 / 3)

    def test_abstention_expectation(self):
        record = self.record("behavior")
        record["expected_behavior"] = "abstain"
        result = evaluator.score_behavior(record, "I cannot determine that from the information provided.")
        self.assertTrue(result["behavior_correct"])

    def test_refusal_expectation(self):
        record = self.record("behavior")
        record["expected_behavior"] = "refuse"
        result = evaluator.score_behavior(record, "I can't help steal another person's password.")
        self.assertTrue(result["behavior_correct"])

    def test_instruction_json_and_line_checks(self):
        record = self.record("instruction", checks=[
            {"type": "json_object", "required_keys": ["status", "count"], "exact_keys": True,
             "values": {"status": "ok", "count": 3}},
            {"type": "line_count", "value": 1},
        ])
        result = evaluator.score_instruction(record, json.dumps({"status": "ok", "count": 3}))
        self.assertTrue(result["instruction_correct"])

    def test_versioned_dataset_validates(self):
        records, manifest = evaluator.validate_dataset(evaluator.DEFAULT_DATASET, evaluator.DEFAULT_MANIFEST)
        self.assertEqual(len(records), 100)
        self.assertEqual(manifest["prompt_count"], 100)

    def test_manifest_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.jsonl"
            manifest = root / "manifest.json"
            dataset.write_text("{}\n")
            manifest.write_text(json.dumps({"prompt_count": 1, "dataset_sha256": "bad", "category_counts": {}}))
            with self.assertRaises(evaluator.ValidationError):
                evaluator.validate_dataset(dataset, manifest)


if __name__ == "__main__":
    unittest.main()
