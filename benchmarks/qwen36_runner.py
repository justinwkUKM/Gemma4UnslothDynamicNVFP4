#!/usr/bin/env python3
"""Probe NVFP4 MoE backends and run only missing Qwen3.6 benchmark work."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
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

from campaigns.common import (  # noqa: E402
    CampaignManifest,
    atomic_write_json,
    canonical_sha256,
    capture_environment,
    sha256_file,
    utc_now,
)


CONFIG_PATH = Path(__file__).with_name("qwen36_config.json")
FAILURE_CLASSES = {
    "dependency_installation",
    "download",
    "model_loading",
    "kernel_selection",
    "oom",
    "benchmark_execution",
    "timeout",
    "budget_exhausted",
}
BACKEND_PATTERNS = {
    "flashinfer_cutedsl": re.compile(r"flashinfer[ _-]*(?:cute[ _-]*dsl|cutedsl)", re.I),
    "flashinfer_trtllm": re.compile(r"flashinfer[ _-]*(?:trt[ _-]*llm|trtllm)", re.I),
    "cutlass": re.compile(r"(?<!flashinfer[ _-])\bcutlass\b", re.I),
}
FALLBACK_PATTERN = re.compile(r"\b(emulation|marlin|triton(?:_unfused)?|cpu[ _-]*offload)\b", re.I)
OOM_PATTERN = re.compile(r"out of memory|cuda oom|memory profiling.*failed", re.I)
UNSUPPORTED_PATTERN = re.compile(
    r"not supported|unsupported|no supported|cannot support|does not support|incompatible|invalid choice|unrecognized arguments",
    re.I,
)
SELECTION_WORDS = re.compile(r"\b(using|selected|selecting|chosen|resolved)\b", re.I)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"models", "backend_probe_order", "server", "workloads"}
    missing = required - value.keys()
    if missing:
        raise ValueError("Qwen config missing: " + ", ".join(sorted(missing)))
    if value["backend_probe_order"] != ["flashinfer_cutedsl", "flashinfer_trtllm", "cutlass"]:
        raise ValueError("backend_probe_order must preserve the approved order")
    return value


def classify_backend_log(log_text: str, backend: str, healthy: bool, exit_code: int | None) -> dict[str, Any]:
    """Require an explicit vLLM selection line; configuration echo is insufficient."""

    lines = log_text.splitlines()
    relevant = [line[-2000:] for line in lines if SELECTION_WORDS.search(line) or UNSUPPORTED_PATTERN.search(line) or OOM_PATTERN.search(line)]
    selected_line = next(
        (line for line in lines if SELECTION_WORDS.search(line) and BACKEND_PATTERNS[backend].search(line)),
        None,
    )
    fallback_line = None
    for line in lines:
        selection_clause = re.split(r"(?:out of )?potential backends", line, maxsplit=1, flags=re.I)[0]
        if SELECTION_WORDS.search(selection_clause) and FALLBACK_PATTERN.search(selection_clause):
            fallback_line = line
            break
    unsupported_line = next((line for line in lines if UNSUPPORTED_PATTERN.search(line)), None)
    oom_line = next((line for line in lines if OOM_PATTERN.search(line)), None)

    if oom_line:
        status, classification, detail = "failed", "oom", oom_line[-2000:]
    elif fallback_line:
        status, classification, detail = "fallback_rejected", "kernel_selection", fallback_line[-2000:]
    elif not healthy:
        status = "unsupported" if unsupported_line else "failed"
        classification = "kernel_selection" if unsupported_line else "model_loading"
        detail = (unsupported_line or f"server exited={exit_code} before health gate")[-2000:]
    elif not selected_line:
        status, classification = "unconfirmed", "kernel_selection"
        detail = "server became healthy but vLLM did not emit an explicit selected/using backend line"
    else:
        status, classification, detail = "supported", None, selected_line[-2000:]
    return {
        "backend": backend,
        "status": status,
        "classification": classification,
        "healthy": healthy,
        "exit_code": exit_code,
        "confirmation_line": selected_line[-2000:] if selected_line else None,
        "detail": detail,
        "relevant_log_lines": relevant[-20:],
        "rank_eligible": status == "supported",
    }


def server_command(args: argparse.Namespace, checkpoint: str, backend: str, config: Mapping[str, Any]) -> list[str]:
    server = config["server"]
    return [
        args.vllm_command,
        "serve",
        checkpoint,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--served-model-name",
        checkpoint,
        "--max-model-len",
        str(server["context_limit"]),
        "--max-num-seqs",
        str(server["max_num_seqs"]),
        "--gpu-memory-utilization",
        str(server["gpu_memory_utilization"]),
        "--linear-backend",
        "auto",
        "--moe-backend",
        backend,
        "--seed",
        str(server["seed"]),
    ]


def wait_for_health(base_url: str, process: subprocess.Popen[Any], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=5) as response:
                if response.status == 200:
                    with urllib.request.urlopen(base_url + "/v1/models", timeout=5) as models:
                        return models.status == 200
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(2)
    return False


def stop_process_group(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 30
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.25)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def gpu_sampler(path: Path, stopped: threading.Event) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("utc,memory_used_mib,memory_total_mib,utilization_gpu,power_w,temperature_c\n")
        while not stopped.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=10,
                ).strip()
                handle.write(f"{utc_now()},{output.replace(', ', ',')}\n")
            except (OSError, subprocess.SubprocessError) as exc:
                handle.write(f"{utc_now()},ERROR,{type(exc).__name__}: {exc}\n")
            handle.flush()
            stopped.wait(2)


def metric(value: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        candidate = value.get(name)
        if isinstance(candidate, (int, float)):
            return float(candidate)
    return None


def normalize_result(
    raw_path: Path,
    *,
    model_id: str,
    checkpoint: str,
    backend: str,
    workload: str,
    repetition: int,
    command: list[str],
    elapsed_seconds: float,
    error: dict[str, Any] | None,
) -> dict[str, Any]:
    try:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw = {}
    failures = int(raw.get("failed_requests", raw.get("num_failed_requests", 0)) or 0)
    if error is None and (not raw or failures):
        error = {
            "classification": "benchmark_execution",
            "detail": "vLLM result missing or contains failed requests",
        }
    return {
        "schema_version": 1,
        "run_id": f"{model_id}__{workload}__{repetition}",
        "model_id": model_id,
        "checkpoint": checkpoint,
        "backend": backend,
        "context_limit": 4096,
        "workload": workload,
        "repetition": repetition,
        "status": "success" if error is None else "error",
        "raw_file": str(raw_path),
        "command": command,
        "wall_seconds": elapsed_seconds,
        "output_tps": metric(raw, "output_throughput"),
        "total_tps": metric(raw, "total_token_throughput"),
        "request_tps": metric(raw, "request_throughput"),
        "ttft_ms": metric(raw, "median_ttft_ms", "mean_ttft_ms"),
        "tpot_ms": metric(raw, "median_tpot_ms", "mean_tpot_ms"),
        "prompt_tokens": metric(raw, "input_tokens", "prompt_tokens"),
        "generated_tokens": metric(raw, "output_tokens", "generated_tokens"),
        "failures": failures,
        "error": error,
    }


def run_benchmark(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    model_id: str,
    checkpoint: str,
    backend: str,
    workload: str,
    repetition: int,
    model_dir: Path,
) -> dict[str, Any]:
    workload_config = config["workloads"][workload]
    run_id = f"{model_id}__{workload}__{repetition}"
    raw_path = model_dir / "raw" / f"{run_id}.json"
    meta_path = model_dir / "normalized" / f"{run_id}.json"
    stdout_path = model_dir / "logs" / f"{run_id}.log"
    if meta_path.exists() and raw_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if existing.get("status") == "success":
            return existing
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.vllm_command,
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--base-url",
        args.base_url,
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        checkpoint,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(workload_config["input_tokens"]),
        "--random-output-len",
        str(workload_config["output_tokens"]),
        "--random-range-ratio",
        "0",
        "--num-prompts",
        str(workload_config["requests"]),
        "--max-concurrency",
        str(workload_config["concurrency"]),
        "--seed",
        str(config["server"]["seed"]),
        "--ignore-eos",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(raw_path.parent),
        "--result-filename",
        raw_path.name,
        "--percentile-metrics",
        "ttft,tpot,e2el",
        "--metric-percentiles",
        "50,90,99",
    ]
    started = time.monotonic()
    error = None
    output = ""
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=args.benchmark_timeout)
        output = completed.stdout + completed.stderr
        if completed.returncode:
            error = {
                "classification": "benchmark_execution",
                "detail": f"exit {completed.returncode}: {completed.stderr[-2000:]}",
            }
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        error = {"classification": "timeout", "detail": "benchmark invocation exceeded timeout"}
    stdout_path.write_text(output, encoding="utf-8")
    record = normalize_result(
        raw_path,
        model_id=model_id,
        checkpoint=checkpoint,
        backend=backend,
        workload=workload,
        repetition=repetition,
        command=command,
        elapsed_seconds=time.monotonic() - started,
        error=error,
    )
    atomic_write_json(meta_path, record)
    return record


def write_report(root: Path, config: Mapping[str, Any], records: list[dict[str, Any]], gates: list[dict[str, Any]], status: str, hourly_rate: float, elapsed: float) -> None:
    lines = [
        "# Qwen3.6 NVFP4 campaign report\n\n",
        f"Generated: {utc_now()}  \n",
        f"Campaign status: **{status}**  \n",
        "Context limit: 4,096 tokens; maximum sequences: 16; seed: 0.  \n",
        f"Elapsed: {elapsed:.1f} seconds; estimated cost: ${hourly_rate * elapsed / 3600:.6f}.\n\n",
        "> Performance and quality are independent scorecards. A model is ranked only after an explicitly confirmed backend gate and every required repetition.\n\n",
        "## Compatibility attempts\n\n",
        "| Model | Backend | Outcome | Healthy | Rank eligible | Detail |\n",
        "| --- | --- | --- | --- | --- | --- |\n",
    ]
    for gate in gates:
        detail = str(gate.get("detail") or "").replace("|", "\\|").replace("\n", " ")[:240]
        lines.append(
            f"| {gate['model_id']} | `{gate['backend']}` | {gate['status']} | "
            f"{gate['healthy']} | {gate['rank_eligible']} | {detail} |\n"
        )
    lines.extend(
        [
            "\n## Performance\n\n",
            "| Model | Workload | Repetitions | Output tok/s | Total tok/s | Request/s | TTFT ms | TPOT ms | Failures |\n",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
        ]
    )
    for model_id in config["models"]:
        for workload in ("interactive", "throughput"):
            selected = [row for row in records if row["model_id"] == model_id and row["workload"] == workload]
            successful = [row for row in selected if row["status"] == "success"]

            def median(name: str) -> str:
                values = [row[name] for row in successful if isinstance(row.get(name), (int, float))]
                return f"{statistics.median(values):.3f}" if values else "n/a"

            if selected:
                lines.append(
                    f"| {model_id} | {workload} | {len(successful)}/{config['workloads'][workload]['repetitions']} | "
                    f"{median('output_tps')} | {median('total_tps')} | {median('request_tps')} | "
                    f"{median('ttft_ms')} | {median('tpot_ms')} | {sum(row['failures'] + (row['status'] != 'success') for row in selected)} |\n"
                )
    lines.extend(
        [
            "\n## Quality\n\n",
            "Quality is run only after a complete performance gate and is written independently to `quality/results-qwen36/` and `quality/summary/qwen36-quality-report.md`.\n",
        ]
    )
    report = root / "summary" / "qwen36-report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("".join(lines), encoding="utf-8")


def selected_models(config: Mapping[str, Any], requested: list[str] | None) -> list[str]:
    if requested:
        unknown = set(requested) - set(config["models"])
        if unknown:
            raise ValueError("unknown model IDs: " + ", ".join(sorted(unknown)))
        return requested
    return [model_id for model_id, entry in config["models"].items() if entry.get("run_by_default")]


def locate_model_revision(hf_home: Path, checkpoint: str) -> str | None:
    cache_name = "models--" + checkpoint.replace("/", "--")
    candidates = sorted((hf_home / "hub" / cache_name / "snapshots").glob("*/config.json"))
    if not candidates:
        candidates = sorted((hf_home / cache_name / "snapshots").glob("*/config.json"))
    return candidates[-1].parent.name if candidates else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--root", type=Path, help="campaign root; defaults to benchmarks/qwen36_<UTC>")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--vllm-command", default=os.environ.get("VLLM_COMMAND", "vllm"))
    parser.add_argument("--hf-home", type=Path, default=Path(os.environ.get("HF_HOME", "/workspace/gemma4-benchmark/cache/huggingface")))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--probe-timeout", type=float, default=2700)
    parser.add_argument("--benchmark-timeout", type=float, default=1200)
    parser.add_argument("--max-campaign-seconds", type=float, default=18000)
    parser.add_argument("--hourly-rate", type=float, default=0.69)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--skip-quality", action="store_true", help="do not launch the separate quality campaign")
    parser.add_argument("--quality-results-dir", type=Path, default=REPO_ROOT / "quality" / "results-qwen36")
    parser.add_argument("--quality-report", type=Path, default=REPO_ROOT / "quality" / "summary" / "qwen36-quality-report.md")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_campaign_seconds <= 0 or args.hourly_rate <= 0:
        print("error: campaign duration and hourly rate must be positive", file=sys.stderr)
        return 2
    try:
        config = load_config(args.config)
        model_ids = selected_models(config, args.models)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not model_ids:
        print("error: Qwen3.6 35B is skipped; pass --models explicitly only to override that decision", file=sys.stderr)
        return 2
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.root = args.root or REPO_ROOT / "benchmarks" / f"qwen36_{timestamp}"
    args.root.mkdir(parents=True, exist_ok=True)
    config_snapshot = args.root / "config" / "qwen36_config.json"
    config_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, config_snapshot)
    args.base_url = f"http://{args.host}:{args.port}"
    started_monotonic = time.monotonic()
    deadline_monotonic = started_monotonic + args.max_campaign_seconds
    started_at = utc_now()
    deadline_at = (datetime.now(timezone.utc) + timedelta(seconds=args.max_campaign_seconds)).isoformat()
    environment = capture_environment(
        extra={"config_sha256": sha256_file(args.config), "runner_sha256": sha256_file(Path(__file__))}
    )
    environment_path = args.root / "environment" / "runtime.json"
    atomic_write_json(environment_path, environment)
    gpu_query = environment.get("gpu", {}).get("nvidia_smi_query")
    gpu_type = str(gpu_query).split(",", 1)[0] if gpu_query else None
    model_versions = {
        model_id: config["models"][model_id].get("revision")
        or locate_model_revision(args.hf_home, config["models"][model_id]["checkpoint"])
        or "unresolved"
        for model_id in model_ids
    }
    manifest = CampaignManifest.create(
        args.root / "campaign.json",
        campaign_id=args.root.name,
        campaign_type="qwen36-performance",
        dataset_versions={"vllm-random": "fixed-512x512-v1"},
        model_versions=model_versions,
        backend={"probe_order": config["backend_probe_order"], "selected": {}},
        context_limit=config["server"]["context_limit"],
        seed=config["server"]["seed"],
        prompt_hash=canonical_sha256(config["workloads"]),
        environment_hash=environment["environment_sha256"],
        gpu_type=gpu_type,
        started_at_utc=started_at,
        deadline_utc=deadline_at,
        hourly_rate_usd=args.hourly_rate,
        config_hash=sha256_file(args.config),
        extra={"historical_27b_policy": "never overwritten; default target is missing 35B only"},
    )
    gates: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    selected_backends: dict[str, str] = {}
    quality_candidates: list[str] = []

    for model_id in model_ids:
        checkpoint = config["models"][model_id]["checkpoint"]
        model_dir = args.root / "results" / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        process: subprocess.Popen[Any] | None = None
        log_handle = None
        sampler_stop: threading.Event | None = None
        sampler_thread: threading.Thread | None = None
        selected_backend = None
        try:
            for attempt_number, backend in enumerate(config["backend_probe_order"], 1):
                if time.monotonic() >= deadline_monotonic:
                    gates.append(
                        {
                            "model_id": model_id,
                            "checkpoint": checkpoint,
                            "backend": backend,
                            "status": "not_attempted",
                            "classification": "budget_exhausted",
                            "healthy": False,
                            "rank_eligible": False,
                            "detail": "campaign deadline reached",
                        }
                    )
                    break
                command = server_command(args, checkpoint, backend, config)
                log_path = model_dir / "logs" / f"backend-{attempt_number}-{backend}.server.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = log_path.open("w", encoding="utf-8")
                attempt_sampler_stop = threading.Event()
                attempt_sampler_thread = threading.Thread(
                    target=gpu_sampler,
                    args=(model_dir / "telemetry" / f"backend-{attempt_number}-{backend}.gpu.csv", attempt_sampler_stop),
                    daemon=True,
                )
                attempt_sampler_thread.start()
                probe_started = time.monotonic()
                try:
                    process = subprocess.Popen(
                        command,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env=os.environ.copy(),
                    )
                    healthy = wait_for_health(
                        args.base_url,
                        process,
                        min(args.probe_timeout, max(0.1, deadline_monotonic - time.monotonic())),
                    )
                except OSError as exc:
                    healthy = False
                    log_handle.write(f"launcher error: {type(exc).__name__}: {exc}\n")
                log_handle.flush()
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                outcome = classify_backend_log(log_text, backend, healthy, process.poll() if process else None)
                outcome.update(
                    {
                        "model_id": model_id,
                        "checkpoint": checkpoint,
                        "attempt": attempt_number,
                        "command": command,
                        "log_file": str(log_path),
                        "startup_seconds": time.monotonic() - probe_started,
                        "attempted_at_utc": utc_now(),
                    }
                )
                gates.append(outcome)
                atomic_write_json(model_dir / "compatibility.json", {"attempts": [g for g in gates if g["model_id"] == model_id]})
                if outcome["rank_eligible"]:
                    selected_backend = backend
                    selected_backends[model_id] = backend
                    sampler_stop = attempt_sampler_stop
                    sampler_thread = attempt_sampler_thread
                    model_versions[model_id] = locate_model_revision(args.hf_home, checkpoint) or "unresolved"
                    break
                stop_process_group(process)
                attempt_sampler_stop.set()
                attempt_sampler_thread.join(5)
                process = None
                log_handle.close()
                log_handle = None

            if selected_backend is None or args.probe_only:
                continue
            smoke = run_benchmark(args, config, model_id, checkpoint, selected_backend, "smoke", 1, model_dir)
            records.append(smoke)
            if smoke["status"] != "success":
                continue
            for workload in ("warmup", "interactive", "throughput"):
                repetitions = config["workloads"][workload]["repetitions"]
                for repetition in range(1, repetitions + 1):
                    if time.monotonic() >= deadline_monotonic:
                        break
                    records.append(
                        run_benchmark(
                            args,
                            config,
                            model_id,
                            checkpoint,
                            selected_backend,
                            workload,
                            repetition,
                            model_dir,
                        )
                    )
            measured = [row for row in records if row["model_id"] == model_id and row["workload"] in {"interactive", "throughput"}]
            expected = sum(config["workloads"][name]["repetitions"] for name in ("interactive", "throughput"))
            if len(measured) == expected and all(row["status"] == "success" for row in measured):
                quality_candidates.append(model_id)
        finally:
            if sampler_stop:
                sampler_stop.set()
            if sampler_thread:
                sampler_thread.join(5)
            stop_process_group(process)
            if log_handle:
                log_handle.close()

    atomic_write_json(args.root / "results" / "normalized.json", records)
    atomic_write_json(args.root / "results" / "compatibility-attempts.json", gates)
    manifest.update_backend({"probe_order": config["backend_probe_order"], "selected": selected_backends})
    manifest.data["model_versions"] = model_versions
    compatibility_passed = len(selected_backends) == len(model_ids)
    expected_measured = len(model_ids) * sum(config["workloads"][name]["repetitions"] for name in ("interactive", "throughput"))
    measured = [row for row in records if row["workload"] in {"interactive", "throughput"}]
    repetitions_passed = len(measured) == expected_measured and all(row["status"] == "success" for row in measured)
    smoke_passed = all(
        any(row["model_id"] == model_id and row["workload"] == "smoke" and row["status"] == "success" for row in records)
        for model_id in model_ids
    ) if not args.probe_only else compatibility_passed
    performance_complete = compatibility_passed and smoke_passed and repetitions_passed and not args.probe_only

    quality_launched = False
    quality_complete = False
    quality_satisfied = args.skip_quality or args.probe_only
    if performance_complete and not args.skip_quality:
        quality_launched = True
        for model_id in quality_candidates:
            quality_command = [
                sys.executable,
                str(REPO_ROOT / "quality" / "runner.py"),
                "--results-dir",
                str(args.quality_results_dir),
                "--models",
                model_id,
                "--moe-backend",
                selected_backends[model_id],
                "--compatibility-gate",
                str(args.root / "results" / model_id / "compatibility.json"),
                "--hourly-rate",
                str(args.hourly_rate),
            ]
            completed = subprocess.run(quality_command, text=True)
            if completed.returncode:
                quality_complete = False
                break
        else:
            evaluator_command = [
                sys.executable,
                str(REPO_ROOT / "quality" / "evaluator.py"),
                "--results-dir",
                str(args.quality_results_dir),
                "--report",
                str(args.quality_report),
                "--models",
                *quality_candidates,
                "--campaign-title",
                "Qwen3.6 quality and factuality report",
            ]
            quality_complete = subprocess.run(evaluator_command, text=True).returncode == 0
            quality_satisfied = quality_complete

    requirements = {
        "backend_gate": compatibility_passed,
        "smoke": smoke_passed,
        "all_required_repetitions": repetitions_passed,
        "quality_complete_or_explicitly_skipped": quality_satisfied,
        "model_revisions_resolved": all(revision != "unresolved" for revision in model_versions.values()),
    }
    complete = performance_complete and quality_satisfied and requirements["model_revisions_resolved"]
    status = "complete" if complete else "partial"
    elapsed = time.monotonic() - started_monotonic
    write_report(args.root, config, records, gates, status, args.hourly_rate, elapsed)
    manifest.data["quality"] = {
        "launched": quality_launched,
        "complete": quality_complete,
        "explicitly_skipped": args.skip_quality,
        "results_dir": str(args.quality_results_dir),
        "report": str(args.quality_report),
    }
    manifest.finish(status, requirements=requirements, artifact_root=args.root)
    print(json.dumps({"status": status, "root": str(args.root), "requirements": requirements}, indent=2))
    return 0 if complete or args.probe_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
