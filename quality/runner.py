#!/usr/bin/env python3
"""Run the versioned quality suite against Gemma checkpoints served by vLLM."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .evaluator import DEFAULT_DATASET, DEFAULT_MANIFEST, DEFAULT_RESULTS, read_jsonl, sha256_file, validate_dataset
except ImportError:  # Direct execution: python quality/runner.py
    from evaluator import DEFAULT_DATASET, DEFAULT_MANIFEST, DEFAULT_RESULTS, read_jsonl, sha256_file, validate_dataset


ROOT = Path(__file__).resolve().parent
MODELS = {
    "gemma-4-E4B-it-NVFP4": "unsloth/gemma-4-E4B-it-NVFP4",
    "gemma-4-12b-it-NVFP4": "unsloth/gemma-4-12b-it-NVFP4",
    "gemma-4-26B-A4B-it-NVFP4": "unsloth/gemma-4-26B-A4B-it-NVFP4",
    "qwen3.6-27B-NVFP4": "unsloth/Qwen3.6-27B-NVFP4",
    "qwen3.6-35B-A3B-NVFP4": "unsloth/Qwen3.6-35B-A3B-NVFP4",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def safe_packages() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("vllm", "torch", "transformers", "flashinfer-python", "nvidia-cutlass-dsl"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def gpu_metadata() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,compute_cap,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=10).strip()
        return {"nvidia_smi_query": output}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"nvidia_smi_error": f"{type(exc).__name__}: {exc}"}


def environment_metadata(dataset_hash: str) -> dict[str, Any]:
    return {
        "captured_at_utc": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": safe_packages(),
        "gpu": gpu_metadata(),
        "dataset_sha256": dataset_hash,
        "runner_sha256": sha256_file(Path(__file__)),
    }


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex((host, port)) != 0


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 10) -> dict[str, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
        return json.loads(body) if body else {}


def wait_for_server(base_url: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = "server did not respond"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=5) as response:
                if response.status == 200:
                    return request_json(base_url + "/v1/models", timeout=5)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise TimeoutError(last_error)


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


def atomic_write_jsonl(path: Path, records: dict[str, dict[str, Any]], order: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for prompt_id in order:
            if prompt_id in records:
                handle.write(json.dumps(records[prompt_id], sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {record["prompt_id"]: record for record in read_jsonl(path) if record.get("prompt_id")}


def locate_model_config(hf_home: Path, checkpoint: str) -> dict[str, Any]:
    cache_name = "models--" + checkpoint.replace("/", "--")
    candidates = sorted((hf_home / "hub" / cache_name / "snapshots").glob("*/config.json"))
    if not candidates:
        candidates = sorted((hf_home / cache_name / "snapshots").glob("*/config.json"))
    if not candidates:
        return {"model_config_path": None, "model_config_sha256": None, "model_revision": None}
    path = candidates[-1]
    return {
        "model_config_path": str(path),
        "model_config_sha256": sha256_file(path),
        "model_revision": path.parent.name,
    }


def generation_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "temperature": 0,
        "top_p": 1,
        "top_k": -1,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
    }


def make_payload(checkpoint: str, prompt: str, settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": checkpoint,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": settings["temperature"],
        "top_p": settings["top_p"],
        "top_k": settings["top_k"],
        "seed": settings["seed"],
        "max_tokens": settings["max_tokens"],
        "stream": False,
    }


def one_attempt(base_url: str, checkpoint: str, prompt: str, settings: dict[str, Any], timeout: float) -> dict[str, Any]:
    started_utc = utc_now()
    started = time.monotonic()
    payload = make_payload(checkpoint, prompt, settings)
    try:
        response = request_json(base_url + "/v1/chat/completions", payload, timeout)
        choices = response.get("choices") or []
        output = choices[0].get("message", {}).get("content", "") if choices else ""
        if not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False)
        return {
            "status": "success",
            "started_at_utc": started_utc,
            "finished_at_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "output_text": output,
            "finish_reason": choices[0].get("finish_reason") if choices else None,
            "usage": response.get("usage", {}),
            "response_id": response.get("id"),
        }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[-2000:]
        classification = "server_error" if exc.code >= 500 else "request_rejected"
        detail = f"HTTP {exc.code}: {body}"
    except TimeoutError as exc:
        classification, detail = "timeout", str(exc)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        classification, detail = "transport_error", f"{type(exc).__name__}: {exc}"
    return {
        "status": "error",
        "started_at_utc": started_utc,
        "finished_at_utc": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "error": {"classification": classification, "detail": detail},
    }


def evaluate_prompt(
    base_url: str,
    model_id: str,
    checkpoint: str,
    item: dict[str, Any],
    settings: dict[str, Any],
    args: argparse.Namespace,
    existing: dict[str, Any] | None,
    common: dict[str, Any],
) -> dict[str, Any]:
    attempts = list(existing.get("attempts", [])) if existing else []
    final: dict[str, Any] = {}
    for _ in range(args.max_retries + 1):
        attempt = one_attempt(base_url, checkpoint, item["prompt"], settings, args.request_timeout)
        attempts.append(attempt)
        final = attempt
        if attempt["status"] == "success":
            break
        if attempt.get("error", {}).get("classification") == "request_rejected":
            break
    return {
        **common,
        "prompt_id": item["id"],
        "category": item["category"],
        "prompt": item["prompt"],
        "status": final.get("status", "error"),
        "output_text": final.get("output_text", ""),
        "finish_reason": final.get("finish_reason"),
        "usage": final.get("usage", {}),
        "elapsed_seconds": final.get("elapsed_seconds"),
        "request_cost_usd": args.hourly_rate * float(final.get("elapsed_seconds") or 0) / 3600,
        "error": final.get("error"),
        "attempts": attempts,
    }


def server_command(args: argparse.Namespace, checkpoint: str) -> list[str]:
    command = [
        args.vllm_command,
        "serve",
        checkpoint,
        "--host", args.host,
        "--port", str(args.port),
        "--served-model-name", checkpoint,
        *( ["--max-model-len", "4096", "--max-num-seqs", "16"]
           if checkpoint.lower().startswith("unsloth/qwen")
           else ["--max-model-len", "8192"] ),
        "--gpu-memory-utilization", "0.90",
        "--linear-backend", "auto",
        "--moe-backend", "flashinfer_cutlass",
        "--seed", str(args.seed),
    ]
    if checkpoint.lower().startswith("unsloth/gemma"):
        command[command.index("--linear-backend"):command.index("--linear-backend")] = ["--reasoning-parser", "gemma4"]
    return command


def record_model_failure(
    model_id: str,
    checkpoint: str,
    items: list[dict[str, Any]],
    output: Path,
    common: dict[str, Any],
    classification: str,
    detail: str,
) -> None:
    records = load_existing(output)
    for item in items:
        if records.get(item["id"], {}).get("status") == "success":
            continue
        records[item["id"]] = {
            **common,
            "prompt_id": item["id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "model_id": model_id,
            "checkpoint": checkpoint,
            "status": "error",
            "output_text": "",
            "usage": {},
            "error": {"classification": classification, "detail": detail},
            "attempts": [],
        }
    atomic_write_jsonl(output, records, [item["id"] for item in items])


def run_model(
    model_id: str,
    checkpoint: str,
    items: list[dict[str, Any]],
    manifest: dict[str, Any],
    environment_file: Path,
    args: argparse.Namespace,
) -> None:
    model_dir = args.results_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    raw_path = model_dir / "raw.jsonl"
    existing = load_existing(raw_path)
    selected_ids = [item["id"] for item in items]
    settings = generation_settings(args)
    settings_hash = canonical_hash(settings)
    command = [] if args.external_server else server_command(args, checkpoint)
    server_hash = canonical_hash(command)
    model_config = locate_model_config(args.hf_home, checkpoint)
    common = {
        "schema_version": 1,
        "model_id": model_id,
        "checkpoint": checkpoint,
        "checkpoint_identifier_sha256": hashlib.sha256(checkpoint.encode()).hexdigest(),
        **model_config,
        "dataset_version": manifest["dataset_version"],
        "dataset_sha256": manifest["dataset_sha256"],
        "generation_settings": settings,
        "generation_settings_sha256": settings_hash,
        "server_config_sha256": server_hash,
        "environment_file": str(environment_file),
    }
    process: subprocess.Popen[Any] | None = None
    log_handle = None
    started = time.monotonic()
    startup_seconds = 0.0
    run_status = "complete"
    started_at_utc = utc_now()
    try:
        if time.monotonic() >= args.campaign_deadline:
            raise RuntimeError("quality campaign budget exhausted before model startup")
        if args.external_server:
            remaining = max(0.1, args.campaign_deadline - time.monotonic())
            wait_for_server(args.base_url, min(args.server_startup_timeout, remaining))
        else:
            if not port_is_free(args.host, args.port):
                raise RuntimeError(f"{args.host}:{args.port} is already in use")
            log_path = model_dir / "server.log"
            log_handle = log_path.open("a", encoding="utf-8")
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=os.environ.copy(),
            )
            startup_started = time.monotonic()
            remaining = max(0.1, args.campaign_deadline - time.monotonic())
            wait_for_server(args.base_url, min(args.server_startup_timeout, remaining))
            startup_seconds = time.monotonic() - startup_started
        for item in items:
            previous = existing.get(item["id"])
            if previous and previous.get("status") == "success":
                continue
            if time.monotonic() >= args.campaign_deadline:
                run_status = "partial"
                existing[item["id"]] = {
                    **common,
                    "prompt_id": item["id"],
                    "category": item["category"],
                    "prompt": item["prompt"],
                    "status": "error",
                    "output_text": "",
                    "usage": {},
                    "error": {
                        "classification": "budget_exhausted",
                        "detail": "quality campaign wall-time or cost limit reached",
                    },
                    "attempts": [],
                }
                atomic_write_jsonl(raw_path, existing, selected_ids)
                continue
            record = evaluate_prompt(args.base_url, model_id, checkpoint, item, settings, args, previous, common)
            existing[item["id"]] = record
            atomic_write_jsonl(raw_path, existing, selected_ids)
            if record["status"] != "success":
                run_status = "partial"
    except TimeoutError as exc:
        run_status = "failed"
        record_model_failure(model_id, checkpoint, items, raw_path, common, "model_loading_timeout", str(exc))
    except (OSError, RuntimeError, urllib.error.URLError, json.JSONDecodeError) as exc:
        run_status = "failed"
        record_model_failure(model_id, checkpoint, items, raw_path, common, "model_loading", f"{type(exc).__name__}: {exc}")
    finally:
        if not args.external_server:
            stop_process_group(process)
        if log_handle:
            log_handle.close()
    active_seconds = time.monotonic() - started
    run_record = {
        "schema_version": 1,
        "model_id": model_id,
        "checkpoint": checkpoint,
        "status": run_status,
        "started_at_utc": started_at_utc,
        "active_wall_seconds": active_seconds,
        "startup_seconds": startup_seconds,
        "hourly_rate_usd": args.hourly_rate,
        "estimated_cost_usd": args.hourly_rate * active_seconds / 3600,
        "prompt_ids": selected_ids,
        "generation_settings": settings,
        "generation_settings_sha256": settings_hash,
        "server_command": command,
        "server_config_sha256": server_hash,
        **model_config,
    }
    (model_dir / "run.json").write_text(json.dumps(run_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--prompt-id", action="append", default=[], help="run only this prompt ID; repeatable")
    parser.add_argument("--limit", type=int, help="run the first N selected prompts for a smoke test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--server-startup-timeout", type=float, default=2700)
    parser.add_argument("--hourly-rate", type=float, default=0.69)
    parser.add_argument("--max-campaign-seconds", type=float, default=7200, help="hard runner wall-time budget")
    parser.add_argument("--max-cost-usd", type=float, default=1.50, help="compute budget at --hourly-rate")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--base-url", help="OpenAI-compatible server URL (defaults to host and port)")
    parser.add_argument("--external-server", action="store_true", help="use an already running server; requires one model")
    parser.add_argument("--vllm-command", default=os.environ.get("VLLM_COMMAND", "vllm"))
    parser.add_argument("--hf-home", type=Path, default=Path(os.environ.get("HF_HOME", "/workspace/gemma4-benchmark/cache/huggingface")))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.external_server and len(args.models) != 1:
        print("error: --external-server requires exactly one --models value", file=sys.stderr)
        return 2
    if not args.base_url:
        args.base_url = f"http://{args.host}:{args.port}"
    if args.limit is not None and args.limit < 1:
        print("error: --limit must be positive", file=sys.stderr)
        return 2
    if args.hourly_rate <= 0 or args.max_campaign_seconds <= 0 or args.max_cost_usd <= 0:
        print("error: hourly rate and campaign budgets must be positive", file=sys.stderr)
        return 2
    try:
        dataset, manifest = validate_dataset(args.dataset, args.manifest)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.prompt_id:
        requested = set(args.prompt_id)
        unknown = requested - {item["id"] for item in dataset}
        if unknown:
            print("error: unknown prompt IDs: " + ", ".join(sorted(unknown)), file=sys.stderr)
            return 2
        dataset = [item for item in dataset if item["id"] in requested]
    if args.limit is not None:
        dataset = dataset[: args.limit]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    environment_dir = ROOT / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    environment_file = environment_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json")
    environment_file.write_text(json.dumps(environment_metadata(manifest["dataset_sha256"]), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    budget_seconds = min(args.max_campaign_seconds, args.max_cost_usd / args.hourly_rate * 3600)
    args.campaign_deadline = time.monotonic() + budget_seconds
    for model_id in args.models:
        run_model(model_id, MODELS[model_id], dataset, manifest, environment_file, args)
    print(json.dumps({"models": args.models, "prompt_ids": [item["id"] for item in dataset], "results_dir": str(args.results_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
