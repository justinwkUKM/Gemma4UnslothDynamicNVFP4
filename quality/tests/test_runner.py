import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runner


class RunnerPersistenceTests(unittest.TestCase):
    def test_atomic_jsonl_preserves_order_and_partial_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            records = {
                "p-002": {"prompt_id": "p-002", "status": "error"},
                "p-001": {"prompt_id": "p-001", "status": "success"},
            }
            runner.atomic_write_jsonl(path, records, ["p-001", "p-002"])
            loaded = runner.read_jsonl(path)
            self.assertEqual([row["prompt_id"] for row in loaded], ["p-001", "p-002"])
            self.assertEqual(loaded[1]["status"], "error")

    def test_generation_payload_is_deterministic(self):
        settings = {"temperature": 0, "top_p": 1, "top_k": -1, "seed": 0, "max_tokens": 256}
        first = runner.make_payload("checkpoint", "prompt", settings)
        second = runner.make_payload("checkpoint", "prompt", settings)
        self.assertEqual(first, second)
        self.assertEqual(first["temperature"], 0)
        self.assertFalse(first["stream"])


if __name__ == "__main__":
    unittest.main()
