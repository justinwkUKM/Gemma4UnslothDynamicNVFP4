#!/usr/bin/env python3
"""Run controlled dense/MoE efficiency arms with matched vLLM workloads."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.qwen36_runner import (  # noqa: E402
    classify_backend_log,
    gpu_sampler,
    run_benchmark,
    stop_process_group,
    wait_for_health,
)
from campaigns.common import (  # noqa: E402
    CampaignManifest,
    atomic_write_json,
    canonical_sha256,
    capture_environment,
    sha256_file,
    utc_now,
)


CONFIG_PATH = Path(__file__).with_name("moe_config.json")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    workload = value.get("workload", {})
    if not value.get("models") or not workload.get("concurrency_levels"):
        raise ValueError("MoE config requires models and concurrency levels")
    if any(level > value["server"]["max_num_seqs"] for level in workload["concurrency_levels"]):
        raise ValueError("concurrency cannot exceed max_num_seqs")
    return value


def locate_model_config(hf_home: Path, checkpoint: str) -> tuple[Path | None, dict[str, Any]]:
    cache_name = "models--" + checkpoint.replace("/", "--")
    candidates = sorted((hf_home / "hub" / cache_name / "snapshots").glob("*/config.json"))
    if not candidates:
        candidates = sorted((hf_home / cache_name / "snapshots").glob("*/config.json"))
    if not candidates:
        return None, {}
    path = candidates[-1]
    return path, json.loads(path.read_text(encoding="utf-8"))


def nested_find(value: Any, names: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
        for child in value.values():
            found = nested_find(child, names)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = nested_find(child, names)
            if found is not None:
                return found
    return None


def model_metadata(model_id: str, entry: Mapping[str, Any], hf_home: Path) -> dict[str, Any]:
    config_path, raw = locate_model_config(hf_home, entry["checkpoint"])
    architecture = entry["architecture"]
    return {
        "model_id": model_id,
        "checkpoint": entry["checkpoint"],
        "architecture": architecture,
        "total_parameters_billions": entry.get("total_parameters_billions"),
        "active_parameters_billions": entry.get("active_parameters_billions"),
        "expert_count": 1 if architecture == "dense" else nested_find(raw, ("num_local_experts", "num_experts", "n_routed_experts")),
        "expert_top_k": 1 if architecture == "dense" else nested_find(raw, ("num_experts_per_tok", "experts_per_token", "top_k")),
        "model_config_path": str(config_path) if config_path else None,
        "model_config_sha256": sha256_file(config_path) if config_path else None,
        "model_revision": config_path.parent.name if config_path else None,
        "parameter_source": entry.get("parameter_source"),
    }


def server_command(args: argparse.Namespace, entry: Mapping[str, Any], routing_telemetry: bool, config: Mapping[str, Any]) -> list[str]:
    server = config["server"]
    command = [
        args.vllm_command,
        "serve",
        entry["checkpoint"],
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--served-model-name",
        entry["checkpoint"],
        "--max-model-len",
        str(server["context_limit"]),
        "--max-num-seqs",
        str(server["max_num_seqs"]),
        "--gpu-memory-utilization",
        str(server["gpu_memory_utilization"]),
        "--reasoning-parser",
        "gemma4",
        "--linear-backend",
        "auto",
        "--moe-backend",
        server["backend"],
        "--seed",
        str(server["seed"]),
    ]
    if routing_telemetry:
        command.append("--enable-return-routed-experts")
    return command


def scrape_metrics(base_url: str) -> str | None:
    try:
        with urllib.request.urlopen(base_url + "/metrics", timeout=10) as response:
            return response.read().decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError):
        return None


def peak_gpu_metrics(path: Path) -> dict[str, float | None]:
    memory: list[float] = []
    utilization: list[float] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
            fields = line.split(",")
            try:
                memory.append(float(fields[1]))
                utilization.append(float(fields[3]))
            except (IndexError, ValueError):
                pass
    return {
        "peak_gpu_memory_mib": max(memory) if memory else None,
        "mean_gpu_utilization_percent": statistics.fmean(utilization) if utilization else None,
    }


def quality_score(results_dir: Path, model_id: str) -> float | None:
    path = results_dir / model_id / "scores.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    score = value.get("aggregate_quality_score")
    return float(score) if isinstance(score, (int, float)) else None


def parse_prometheus_moe_metrics(text: str | None) -> dict[str, float]:
    """Preserve exact numeric MoE/routing/dispatch metrics exposed by vLLM."""

    values: dict[str, float] = {}
    if not text:
        return values
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name_and_labels, _, raw_value = line.rpartition(" ")
        metric_name = name_and_labels.split("{", 1)[0]
        if not any(term in metric_name.lower() for term in ("moe", "routing", "dispatch", "expert")):
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        values[metric_name] = values.get(metric_name, 0.0) + value
    return dict(sorted(values.items()))


def arm_summary(
    records: list[dict[str, Any]],
    metadata: Mapping[str, Any],
    *,
    model_id: str,
    routing_telemetry: bool,
    concurrency: int,
    hourly_rate: float,
    quality: float | None,
    gpu: Mapping[str, float | None],
    routing_metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    selected = [
        row
        for row in records
        if row["model_id"] == model_id
        and row.get("routing_telemetry") is routing_telemetry
        and row.get("concurrency") == concurrency
    ]

    def median(name: str) -> float | None:
        values = [float(row[name]) for row in selected if row["status"] == "success" and isinstance(row.get(name), (int, float))]
        return statistics.median(values) if values else None

    output_tps = median("output_tps")
    total_params = metadata.get("total_parameters_billions")
    active_params = metadata.get("active_parameters_billions")
    routing_metrics = dict(routing_metrics or {})
    exposed_fraction = next(
        (value for name, value in routing_metrics.items() if name.lower().endswith("dispatch_overhead_fraction")),
        None,
    )
    return {
        "model_id": model_id,
        "architecture": metadata.get("architecture"),
        "backend": selected[0]["backend"] if selected else None,
        "context_limit": 4096,
        "routing_telemetry": routing_telemetry,
        "concurrency": concurrency,
        "repetitions_complete": sum(row["status"] == "success" for row in selected),
        "output_tps": output_tps,
        "total_tps": median("total_tps"),
        "request_tps": median("request_tps"),
        "ttft_ms": median("ttft_ms"),
        "tpot_ms": median("tpot_ms"),
        "total_parameters_billions": total_params,
        "active_parameters_billions": active_params,
        "expert_count": metadata.get("expert_count"),
        "expert_top_k": metadata.get("expert_top_k"),
        "quality_score": quality,
        "quality_per_active_billion": quality / active_params if quality is not None and active_params else None,
        "quality_per_total_billion": quality / total_params if quality is not None and total_params else None,
        "tokens_per_dollar": output_tps * 3600 / hourly_rate if output_tps is not None else None,
        "quality_per_dollar_per_gpu_hour": quality / hourly_rate if quality is not None else None,
        "peak_gpu_memory_mib": gpu.get("peak_gpu_memory_mib"),
        "mean_gpu_utilization_percent": gpu.get("mean_gpu_utilization_percent"),
        "routing_dispatch_overhead_fraction": exposed_fraction,
        "routing_metrics": routing_metrics,
        "routing_metric_status": "exposed" if routing_metrics else "unavailable in runtime metrics",
        "raw_record_ids": [row["run_id"] for row in selected],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row if key != "raw_record_ids"})
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row.get(key), sort_keys=True) if isinstance(row.get(key), (dict, list)) else row.get(key)
                for key in fields
            })
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_report(path: Path, status: str, rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    lines = [
        "# MoE efficiency report\n\n",
        f"Generated: {utc_now()}  \n",
        f"Campaign status: **{status}**\n\n",
        "> Efficiency and output-equivalence quality are reported independently. All table values link through `raw_record_ids` in `moe-summary.json` to normalized and raw measurements.\n\n",
        "| Model | Type | Routing telemetry | Concurrency | Reps | Output tok/s | TTFT ms | TPOT ms | Quality | Tok/$ |\n",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    ]
    for row in rows:
        fmt = lambda name: f"{row[name]:.3f}" if isinstance(row.get(name), (int, float)) else "n/a"
        lines.append(
            f"| {row['model_id']} | {row['architecture']} | {row['routing_telemetry']} | {row['concurrency']} | "
            f"{row['repetitions_complete']} | {fmt('output_tps')} | {fmt('ttft_ms')} | {fmt('tpot_ms')} | "
            f"{fmt('quality_score')} | {fmt('tokens_per_dollar')} |\n"
        )
    lines.extend(["\n## Failures\n\n", "```json\n", json.dumps(failures, indent=2), "\n```\n"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--vllm-command", default=os.environ.get("VLLM_COMMAND", "vllm"))
    parser.add_argument("--hf-home", type=Path, default=Path(os.environ.get("HF_HOME", "/workspace/gemma4-benchmark/cache/huggingface")))
    parser.add_argument("--quality-results-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=float, default=2700)
    parser.add_argument("--benchmark-timeout", type=float, default=1200)
    parser.add_argument("--max-campaign-seconds", type=float, default=18000)
    parser.add_argument("--hourly-rate", type=float, default=0.69)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    model_ids = args.models or list(config["models"])
    unknown = set(model_ids) - set(config["models"])
    if unknown or args.hourly_rate <= 0 or args.max_campaign_seconds <= 0:
        print("error: invalid models or budget", file=sys.stderr)
        return 2
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.root = args.root or REPO_ROOT / "moe" / timestamp
    args.root.mkdir(parents=True, exist_ok=True)
    config_snapshot = args.root / "config" / "moe_config.json"
    config_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, config_snapshot)
    args.base_url = f"http://{args.host}:{args.port}"
    started = time.monotonic()
    deadline = started + args.max_campaign_seconds
    environment = capture_environment(
        extra={"config_sha256": sha256_file(args.config), "runner_sha256": sha256_file(Path(__file__))}
    )
    atomic_write_json(args.root / "environment" / "runtime.json", environment)
    metadata = {model_id: model_metadata(model_id, config["models"][model_id], args.hf_home) for model_id in model_ids}
    atomic_write_json(args.root / "model-metadata.json", metadata)
    gpu_query = environment.get("gpu", {}).get("nvidia_smi_query")
    manifest = CampaignManifest.create(
        args.root / "campaign.json",
        campaign_id=f"moe-{timestamp}",
        campaign_type="moe-efficiency",
        dataset_versions={"vllm-random": "matched-512x256-v1", "quality": "gemma-quality-v1.0.0"},
        model_versions={key: value.get("model_revision") for key, value in metadata.items()},
        backend=config["server"]["backend"],
        context_limit=config["server"]["context_limit"],
        seed=config["server"]["seed"],
        prompt_hash=canonical_sha256(config["workload"]),
        environment_hash=environment["environment_sha256"],
        gpu_type=str(gpu_query).split(",", 1)[0] if gpu_query else None,
        started_at_utc=utc_now(),
        deadline_utc=(datetime.now(timezone.utc) + timedelta(seconds=args.max_campaign_seconds)).isoformat(),
        hourly_rate_usd=args.hourly_rate,
        config_hash=sha256_file(args.config),
    )
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for model_id in model_ids:
        entry = config["models"][model_id]
        telemetry_arms = config["routing_telemetry_arms"] if entry["architecture"] == "moe" else [False]
        for routing_telemetry in telemetry_arms:
            if time.monotonic() >= deadline:
                failures.append({"model_id": model_id, "classification": "budget_exhausted"})
                break
            arm_name = f"routing-{'on' if routing_telemetry else 'off'}"
            arm_dir = args.root / "arms" / model_id / arm_name
            log_path = arm_dir / "server.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = server_command(args, entry, routing_telemetry, config)
            process = None
            log_handle = log_path.open("w", encoding="utf-8")
            sampler_stop = threading.Event()
            gpu_path = arm_dir / "gpu.csv"
            sampler = threading.Thread(target=gpu_sampler, args=(gpu_path, sampler_stop), daemon=True)
            routing_metrics: dict[str, float] = {}
            try:
                sampler.start()
                process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True, env=os.environ.copy())
                healthy = wait_for_health(args.base_url, process, min(args.startup_timeout, max(0.1, deadline - time.monotonic())))
                log_handle.flush()
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                if entry["architecture"] == "moe":
                    gate = classify_backend_log(log_text, config["server"]["backend"], healthy, process.poll())
                else:
                    forbidden = any(term in log_text.lower() for term in ("selected emulation", "using marlin", "cpu offload"))
                    gate = {"rank_eligible": healthy and not forbidden, "status": "supported" if healthy and not forbidden else "failed", "detail": "dense health/fallback gate"}
                atomic_write_json(arm_dir / "compatibility.json", {**gate, "command": command})
                if not gate["rank_eligible"]:
                    failures.append({"model_id": model_id, "arm": arm_name, "classification": "kernel_selection", "detail": gate})
                    continue
                metadata[model_id] = model_metadata(model_id, entry, args.hf_home)
                atomic_write_json(args.root / "model-metadata.json", metadata)
                for concurrency in config["workload"]["concurrency_levels"]:
                    workload_name = f"c{concurrency}"
                    adapter_config = {
                        "server": config["server"],
                        "workloads": {
                            workload_name: {
                                "input_tokens": config["workload"]["input_tokens"],
                                "output_tokens": config["workload"]["output_tokens"],
                                "requests": config["workload"]["requests"],
                                "concurrency": concurrency,
                            }
                        },
                    }
                    for repetition in range(1, config["workload"]["repetitions"] + 1):
                        record = run_benchmark(
                            args,
                            adapter_config,
                            model_id,
                            entry["checkpoint"],
                            config["server"]["backend"],
                            workload_name,
                            repetition,
                            arm_dir,
                        )
                        record.update({"routing_telemetry": routing_telemetry, "concurrency": concurrency, "arm": arm_name})
                        records.append(record)
                metrics = scrape_metrics(args.base_url)
                if metrics is not None:
                    (arm_dir / "vllm-metrics.prom").write_text(metrics, encoding="utf-8")
                    routing_metrics = parse_prometheus_moe_metrics(metrics)
                    atomic_write_json(arm_dir / "routing-metrics.json", routing_metrics)
            except OSError as exc:
                failures.append({"model_id": model_id, "arm": arm_name, "classification": "model_loading", "detail": str(exc)})
            finally:
                sampler_stop.set()
                if sampler.is_alive():
                    sampler.join(5)
                stop_process_group(process)
                log_handle.close()
            gpu = peak_gpu_metrics(gpu_path)
            for concurrency in config["workload"]["concurrency_levels"]:
                summaries.append(
                    arm_summary(
                        records,
                        metadata[model_id],
                        model_id=model_id,
                        routing_telemetry=routing_telemetry,
                        concurrency=concurrency,
                        hourly_rate=args.hourly_rate,
                        quality=quality_score(args.quality_results_dir, model_id),
                        gpu=gpu,
                        routing_metrics=routing_metrics,
                    )
                )

    atomic_write_json(args.root / "normalized.json", records)
    atomic_write_json(args.root / "summary" / "moe-summary.json", summaries)
    write_csv(args.root / "summary" / "moe-summary.csv", summaries)
    expected_arms = sum(2 if config["models"][model]["architecture"] == "moe" else 1 for model in model_ids)
    expected_records = expected_arms * len(config["workload"]["concurrency_levels"]) * config["workload"]["repetitions"]
    metadata_complete = all(
        value.get("total_parameters_billions") is not None
        and value.get("active_parameters_billions") is not None
        and value.get("expert_count") is not None
        and value.get("expert_top_k") is not None
        and value.get("model_revision") is not None
        for value in metadata.values()
    )
    manifest.data["model_versions"] = {
        model_id: value.get("model_revision") or "unresolved"
        for model_id, value in metadata.items()
    }
    complete = len(records) == expected_records and all(row["status"] == "success" for row in records) and not failures and metadata_complete
    status = "complete" if complete else "partial"
    write_report(args.root / "summary" / "moe-report.md", status, summaries, failures)
    manifest.finish(
        status,
        requirements={
            "identical_context_and_workload": True,
            "all_arms_complete": complete,
            "no_hidden_fallback": not any(item.get("classification") == "kernel_selection" for item in failures),
            "raw_links_present": all(bool(row["raw_record_ids"]) for row in summaries),
            "model_metadata_complete": metadata_complete,
        },
        artifact_root=args.root,
    )
    print(json.dumps({"status": status, "root": str(args.root), "records": len(records)}, indent=2))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
