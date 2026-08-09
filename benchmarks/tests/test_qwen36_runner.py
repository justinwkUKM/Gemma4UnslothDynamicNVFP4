import tempfile
import unittest
from pathlib import Path

from benchmarks import qwen36_runner


class QwenBackendGateTests(unittest.TestCase):
    def test_probe_order_is_fixed(self):
        config = qwen36_runner.load_config()
        self.assertEqual(config["backend_probe_order"], ["flashinfer_cutedsl", "flashinfer_trtllm", "cutlass"])

    def test_healthy_server_requires_explicit_backend_confirmation(self):
        verdict = qwen36_runner.classify_backend_log("moe_backend='flashinfer_cutedsl'", "flashinfer_cutedsl", True, None)
        self.assertEqual(verdict["status"], "unconfirmed")
        self.assertFalse(verdict["rank_eligible"])

    def test_selected_backend_passes(self):
        log = "Using FlashInfer CuteDSL NVFP4 MoE backend out of potential backends: [FlashInfer CuteDSL, Triton, Emulation]"
        verdict = qwen36_runner.classify_backend_log(log, "flashinfer_cutedsl", True, None)
        self.assertEqual(verdict["status"], "supported")
        self.assertTrue(verdict["rank_eligible"])

    def test_emulation_fallback_is_never_ranked(self):
        log = "Selected emulation MoE backend because requested kernel is unsupported"
        verdict = qwen36_runner.classify_backend_log(log, "cutlass", True, None)
        self.assertEqual(verdict["status"], "fallback_rejected")
        self.assertFalse(verdict["rank_eligible"])

    def test_qwen_35b_is_skipped_by_default(self):
        config = qwen36_runner.load_config()
        self.assertEqual(qwen36_runner.selected_models(config, None), [])
        self.assertEqual(config["models"]["qwen3.6-35B-A3B-NVFP4"]["historical_status"], "skipped_by_user")


if __name__ == "__main__":
    unittest.main()
