#!/usr/bin/env python3
"""Replay canonical telemetry through an OpenAI-compatible reasoning model."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from campaigns.common import (  # noqa: E402
    CampaignManifest,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_sha256,
    capture_environment,
    read_jsonl,
    sha256_file,
    utc_now,
)
from benchmarks.qwen36_runner import gpu_sampler  # noqa: E402
from quality.security.context import MODES, bound_investigations, build_investigations, model_messages  # noqa: E402
from quality.security.contract import ContractError, parse_model_json, validate_analysis  # noqa: E402
from quality.security.state import IncidentState, InvestigationTools  # noqa: E402


LABEL_KEYS = {"label", "labels", "ground_truth", "attack", "attack_name", "malicious", "is_malicious"}
SECURITY_MODELS = {
    "gemma-4-E4B-it-NVFP4": "unsloth/gemma-4-E4B-it-NVFP4",
    "gemma-4-12b-it-NVFP4": "unsloth/gemma-4-12b-it-NVFP4",
    "gemma-4-26B-A4B-it-NVFP4": "unsloth/gemma-4-26B-A4B-it-NVFP4",
}


def _claim_schema(text_key: str, *, confidence: bool = False) -> dict[str, Any]:
    properties: dict[str, Any] = {
        text_key: {"type": "string", "maxLength": 120},
        "evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
        },
    }
    required = [text_key, "evidence_ids"]
    if confidence:
        properties["confidence"] = {"type": "number", "minimum": 0, "maximum": 1}
        required.append("confidence")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


ANALYSIS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["benign", "suspicious", "malicious", "insufficient_evidence"]},
        "risk_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "observations": {"type": "array", "items": _claim_schema("summary"), "maxItems": 3},
        "hypotheses": {"type": "array", "items": _claim_schema("description", confidence=True), "maxItems": 2},
        "attack_techniques": {"type": "array", "items": _claim_schema("technique_id"), "maxItems": 2},
        "related_entities": {"type": "array", "items": {"type": "string", "maxLength": 120}, "maxItems": 5},
        "recommendations": {"type": "array", "items": _claim_schema("action"), "maxItems": 2},
        "missing_information": {"type": "array", "items": {"type": "string", "maxLength": 120}, "maxItems": 2},
        "predicted_next_actions": {"type": "array", "items": _claim_schema("action", confidence=True), "maxItems": 2},
    },
    "required": [
        "status", "risk_score", "confidence", "observations", "hypotheses", "attack_techniques",
        "related_entities", "recommendations", "missing_information", "predicted_next_actions",
    ],
    "additionalProperties": False,
}


def assert_inference_safe(events: list[dict[str, Any]], *, track: str) -> None:
    for event in events:
        leaked = LABEL_KEYS & {key.lower() for key in event}
        leaked |= LABEL_KEYS & {key.lower() for key in event.get("attributes", {})}
        if leaked:
            raise ValueError(f"event {event.get('event_id')} contains inference label fields: {sorted(leaked)}")
        if track == "public" and event.get("public_data_anonymized") is not True:
            raise ValueError(f"public event {event.get('event_id')} was not anonymized")


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["investigation_id"]: row for row in read_jsonl(path)}


def request_stream(base_url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks: list[str] = []
    usage: dict[str, Any] = {}
    first_token_seconds = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            value = json.loads(data)
            if value.get("usage"):
                usage = value["usage"]
            choices = value.get("choices") or []
            content = choices[0].get("delta", {}).get("content") if choices else None
            if content:
                if first_token_seconds is None:
                    first_token_seconds = time.monotonic() - started
                chunks.append(str(content))
    elapsed = time.monotonic() - started
    completion_tokens = usage.get("completion_tokens")
    tpot = None
    if first_token_seconds is not None and isinstance(completion_tokens, (int, float)) and completion_tokens > 1:
        tpot = max(0.0, elapsed - first_token_seconds) / (completion_tokens - 1)
    return {
        "output_text": "".join(chunks),
        "latency_seconds": elapsed,
        "ttft_seconds": first_token_seconds,
        "tpot_seconds": tpot,
        "usage": usage,
    }


def payload(model: str, messages: list[dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "security_analysis",
                "strict": True,
                "schema": ANALYSIS_JSON_SCHEMA,
            },
        },
    }


def one_attempt(job: dict[str, Any], args: argparse.Namespace, tools: InvestigationTools | None) -> dict[str, Any]:
    started_at = utc_now()
    response: dict[str, Any] = {}
    try:
        response = request_stream(args.base_url, payload(args.model, model_messages(job), args), args.request_timeout)
        preliminary = parse_model_json(response["output_text"])
        tool_requests = preliminary.get("tool_requests", []) if job["mode"] == "tool_using" else []
        tool_results = []
        if tool_requests:
            if tools is None:
                raise ContractError("tool requests are unavailable")
            for request in tool_requests[: args.max_tool_calls]:
                if not isinstance(request, dict):
                    raise ContractError("tool request must be an object")
                name, entity = request.get("name"), request.get("entity")
                if not isinstance(name, str) or not isinstance(entity, str):
                    raise ContractError("tool request requires name and entity")
                tool_results.append({"name": name, "entity": entity, "events": tools.call(name, entity)})
            followup_job = {**job, "tool_results": tool_results}
            followup = request_stream(args.base_url, payload(args.model, model_messages(followup_job), args), args.request_timeout)
            response = {
                **followup,
                "latency_seconds": response["latency_seconds"] + followup["latency_seconds"],
                "tool_results": tool_results,
                "preliminary_output": response["output_text"],
            }
            preliminary = parse_model_json(response["output_text"])
        verdict = validate_analysis(preliminary, set(job["event_ids"]) | {event["event_id"] for result in tool_results for event in result["events"]})
        if not verdict["valid"]:
            raise ContractError("unsupported evidence IDs: " + ", ".join(verdict["unsupported_evidence_ids"]))
        return {
            "status": "success",
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            **response,
            "analysis": preliminary,
            "contract_verdict": verdict,
        }
    except urllib.error.HTTPError as exc:
        detail = f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[-2000:]}"
        classification = "server_error" if exc.code >= 500 else "request_rejected"
    except TimeoutError as exc:
        classification, detail = "timeout", str(exc)
    except (OSError, urllib.error.URLError, json.JSONDecodeError, ContractError, ValueError) as exc:
        classification = "contract_error" if isinstance(exc, (json.JSONDecodeError, ContractError)) else "transport_error"
        detail = f"{type(exc).__name__}: {exc}"
    return {
        "status": "error",
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "output_text": response.get("output_text", ""),
        "latency_seconds": response.get("latency_seconds"),
        "ttft_seconds": response.get("ttft_seconds"),
        "tpot_seconds": response.get("tpot_seconds"),
        "usage": response.get("usage", {}),
        "error": {"classification": classification, "detail": detail},
    }


def execute_job(job: dict[str, Any], args: argparse.Namespace, tools: InvestigationTools | None, previous: dict[str, Any] | None) -> dict[str, Any]:
    attempts = list(previous.get("attempts", [])) if previous else []
    final: dict[str, Any] = {}
    for _ in range(args.max_retries + 1):
        final = one_attempt(job, args, tools)
        attempts.append(final)
        if final["status"] == "success" or final.get("error", {}).get("classification") in {"request_rejected", "contract_error"}:
            break
    usage = final.get("usage", {})
    tool_event_ids = [
        event["event_id"]
        for result in final.get("tool_results", [])
        for event in result.get("events", [])
    ]
    return {
        "schema_version": 1,
        "investigation_id": job["investigation_id"],
        "scenario_id": args.scenario_id,
        "track": args.track,
        "mode": job["mode"],
        "analysis_timestamp": job["analysis_timestamp"],
        "context_event_ids": list(dict.fromkeys(job["event_ids"] + tool_event_ids)),
        "incident_state_input": job.get("incident_state"),
        "context_sha256": canonical_sha256(job),
        "model": args.model,
        "replay_speed": args.replay_speed,
        "concurrent_investigations": args.concurrency,
        "hourly_rate_usd": args.hourly_rate,
        "status": final.get("status", "error"),
        "output_text": final.get("output_text", ""),
        "analysis": final.get("analysis"),
        "latency_seconds": final.get("latency_seconds"),
        "ttft_seconds": final.get("ttft_seconds"),
        "tpot_seconds": final.get("tpot_seconds"),
        "usage": usage,
        "request_cost_usd": args.hourly_rate * float(final.get("latency_seconds") or 0) / 3600,
        "error": final.get("error"),
        "contract_verdict": final.get("contract_verdict"),
        "tool_results": final.get("tool_results", []),
        "attempts": attempts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True, help="canonical, inference-safe JSONL")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--track", choices=("public", "unseen"), required=True)
    parser.add_argument("--mode", choices=MODES, action="append", dest="modes")
    parser.add_argument("--window-seconds", type=float, default=30)
    parser.add_argument("--replay-speed", type=float, default=1)
    parser.add_argument("--model", choices=tuple(SECURITY_MODELS.values()), required=True)
    parser.add_argument("--model-revision", default="unresolved")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--context-limit", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-events-per-context", type=int, default=20)
    parser.add_argument("--max-prompt-bytes", type=int, default=8000)
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-tool-calls", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--hourly-rate", type=float, default=0.69)
    parser.add_argument("--max-campaign-seconds", type=float, default=7200)
    parser.add_argument(
        "--unscored-smoke",
        action="store_true",
        help="run inference but force a partial manifest because benchmark labels/controls are not ready",
    )
    parser.add_argument("--dry-run", action="store_true", help="write contexts without model inference")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.modes = args.modes or ["triggered", "stateful"]
    if (
        args.concurrency < 1 or args.max_campaign_seconds <= 0 or args.hourly_rate <= 0
        or args.max_events_per_context < 1 or args.max_prompt_bytes < 1
    ):
        print("error: concurrency and budgets must be positive", file=sys.stderr)
        return 2
    try:
        events = read_jsonl(args.events)
        assert_inference_safe(events, track=args.track)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.results_dir.mkdir(parents=True, exist_ok=True)
    environment = capture_environment(extra={"events_sha256": sha256_file(args.events)})
    atomic_write_json(args.results_dir / "environment.json", environment)
    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run_config["runner_sha256"] = sha256_file(Path(__file__))
    atomic_write_json(args.results_dir / "config.json", run_config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = CampaignManifest.create(
        args.results_dir / "campaign.json",
        campaign_id=f"security-{args.scenario_id}-{timestamp}",
        campaign_type="security-reasoning",
        dataset_versions={args.dataset_version: sha256_file(args.events)},
        model_versions={args.model: args.model_revision},
        backend="openai-compatible-streaming",
        context_limit=args.context_limit,
        seed=args.seed,
        prompt_hash=canonical_sha256({"modes": args.modes, "window": args.window_seconds, "events": sha256_file(args.events)}),
        environment_hash=environment["environment_sha256"],
        gpu_type=(str(environment.get("gpu", {}).get("nvidia_smi_query")).split(",", 1)[0] if environment.get("gpu", {}).get("nvidia_smi_query") else None),
        started_at_utc=utc_now(),
        deadline_utc=(datetime.now(timezone.utc) + timedelta(seconds=args.max_campaign_seconds)).isoformat(),
        hourly_rate_usd=args.hourly_rate,
        config_hash=canonical_sha256(run_config),
        extra={
            "track": args.track,
            "labels_available_to_model": False,
            "public_data_anonymized": args.track == "public",
            "unscored_smoke": args.unscored_smoke,
        },
    )
    jobs = []
    states: dict[str, IncidentState] = {}
    for mode in args.modes:
        jobs.extend(build_investigations(events, mode=mode, window_seconds=args.window_seconds, state_by_entity=states))
    jobs = bound_investigations(
        jobs,
        max_events=args.max_events_per_context,
        max_prompt_bytes=args.max_prompt_bytes,
    )
    atomic_write_jsonl(args.results_dir / "contexts.jsonl", jobs)
    if args.dry_run:
        manifest.finish(
            "partial",
            requirements={"inference_safe": True, "all_investigations_complete": False},
            artifact_root=args.results_dir,
            detail="dry run generated contexts only",
        )
        print(json.dumps({"status": "partial", "contexts": len(jobs), "dry_run": True}, indent=2))
        return 0

    output_path = args.results_dir / "raw.jsonl"
    existing = load_existing(output_path)
    tools = InvestigationTools(events)
    deadline = time.monotonic() + args.max_campaign_seconds
    sampler_stop = threading.Event()
    sampler_thread = threading.Thread(
        target=gpu_sampler,
        args=(args.results_dir / "gpu.csv", sampler_stop),
        daemon=True,
    )
    sampler_thread.start()
    pending = [job for job in jobs if existing.get(job["investigation_id"], {}).get("status") != "success"]

    # Rebuild incident memory from completed records before resuming a partial
    # stateful stream; successful investigations are never requested again.
    for job in jobs:
        record = existing.get(job["investigation_id"])
        if not record or record.get("status") != "success" or job["mode"] not in {"stateful", "tool_using"}:
            continue
        entity = job["events"][0]["entity_id"]
        state = states.setdefault(entity, IncidentState(f"incident-{entity}", entity))
        state.update(record["analysis"], set(record.get("context_event_ids", [])))

    def persist() -> None:
        atomic_write_jsonl(output_path, (existing[job["investigation_id"]] for job in jobs if job["investigation_id"] in existing))

    if args.concurrency == 1 or any(job["mode"] in {"stateful", "tool_using"} for job in pending):
        for job in pending:
            if time.monotonic() >= deadline:
                break
            if job["mode"] in {"stateful", "tool_using"}:
                entity = job["events"][0]["entity_id"]
                state = states.setdefault(entity, IncidentState(f"incident-{entity}", entity))
                job = {**job, "incident_state": state.as_dict()}
            record = execute_job(job, args, tools if job["mode"] == "tool_using" else None, existing.get(job["investigation_id"]))
            existing[job["investigation_id"]] = record
            if record["status"] == "success" and job["mode"] in {"stateful", "tool_using"}:
                entity = job["events"][0]["entity_id"]
                state = states.setdefault(entity, IncidentState(f"incident-{entity}", entity))
                state.update(record["analysis"], set(record["context_event_ids"]))
            persist()
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(execute_job, job, args, None, existing.get(job["investigation_id"])): job for job in pending}
            for future in as_completed(futures):
                record = future.result()
                existing[record["investigation_id"]] = record
                persist()
    sampler_stop.set()
    sampler_thread.join(5)
    requests_complete = len(existing) == len(jobs) and all(record.get("status") == "success" for record in existing.values())
    contracts_complete = all((record.get("contract_verdict") or {}).get("valid") is True for record in existing.values())
    model_revision_resolved = args.model_revision != "unresolved"
    execution_succeeded = requests_complete and contracts_complete and model_revision_resolved
    complete = execution_succeeded and not args.unscored_smoke
    manifest.finish(
        "complete" if complete else "partial",
        requirements={
            "inference_safe": True,
            "all_investigations_complete": requests_complete,
            "all_claims_contract_valid": contracts_complete,
            "model_revision_resolved": model_revision_resolved,
            "scored_dataset_ready": not args.unscored_smoke,
        },
        artifact_root=args.results_dir,
        detail=(
            "unscored smoke: dataset controls and event-level ground truth are not ready"
            if args.unscored_smoke
            else None
        ),
    )
    print(json.dumps({"status": "complete" if complete else "partial", "investigations": len(jobs), "results": str(output_path)}, indent=2))
    return 0 if execution_succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
