import unittest

from quality import moe_runner


class MoeRunnerTests(unittest.TestCase):
    def test_matched_workload_is_within_context(self):
        config = moe_runner.load_config()
        total = config["workload"]["input_tokens"] + config["workload"]["output_tokens"]
        self.assertLessEqual(total, config["server"]["context_limit"])
        defaults = [model for model, value in config["models"].items() if value.get("run_by_default")]
        self.assertEqual(defaults, [
            "gemma-4-E4B-it-NVFP4",
            "gemma-4-12b-it-NVFP4",
            "gemma-4-26B-A4B-it-NVFP4",
        ])
        self.assertTrue(config["dense_baseline_required"])

    def test_parameter_normalization_keeps_active_and_total_separate(self):
        metadata = {
            "architecture": "moe",
            "total_parameters_billions": 26.0,
            "active_parameters_billions": 4.0,
            "expert_count": 8,
            "expert_top_k": 2,
        }
        records = [{
            "model_id": "m", "routing_telemetry": False, "concurrency": 1,
            "status": "success", "backend": "cutlass", "run_id": "r1",
            "output_tps": 100.0, "total_tps": 200.0, "request_tps": 1.0,
            "ttft_ms": 10.0, "tpot_ms": 5.0,
        }]
        row = moe_runner.arm_summary(
            records, metadata, model_id="m", routing_telemetry=False, concurrency=1,
            hourly_rate=1.0, quality=0.8, gpu={},
        )
        self.assertAlmostEqual(row["quality_per_active_billion"], 0.2)
        self.assertNotEqual(row["quality_per_active_billion"], row["quality_per_total_billion"])
        self.assertEqual(row["raw_record_ids"], ["r1"])

    def test_routing_metrics_preserve_exposed_values(self):
        text = "# HELP x x\nvllm:moe_dispatch_seconds_total 1.5\nvllm:unrelated 9\nvllm:expert_tokens_total{expert=\"1\"} 3\n"
        parsed = moe_runner.parse_prometheus_moe_metrics(text)
        self.assertEqual(parsed["vllm:moe_dispatch_seconds_total"], 1.5)
        self.assertEqual(parsed["vllm:expert_tokens_total"], 3.0)
        self.assertNotIn("vllm:unrelated", parsed)


if __name__ == "__main__":
    unittest.main()
