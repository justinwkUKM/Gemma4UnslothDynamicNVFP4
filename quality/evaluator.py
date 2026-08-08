#!/usr/bin/env python3
"""Deterministic reference-based evaluator for the Gemma quality campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "dataset_v1.jsonl"
DEFAULT_MANIFEST = ROOT / "dataset_manifest.json"
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_REPORT = ROOT / "summary" / "quality-report.md"
EXPECTED_MODELS = (
    "gemma-4-E4B-it-NVFP4",
    "gemma-4-12b-it-NVFP4",
    "gemma-4-26B-A4B-it-NVFP4",
)
FACTUAL_CATEGORIES = {"closed_book_factual_qa", "multi_hop_factual"}
SECRET_PATTERNS = (
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgh[opusr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
)


class ValidationError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValidationError(f"{path}:{line_number}: record must be an object")
            records.append(value)
    return records


def validate_dataset(dataset_path: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = read_jsonl(dataset_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    expected_count = manifest.get("prompt_count")
    if len(records) != expected_count:
        errors.append(f"prompt count is {len(records)}, manifest says {expected_count}")
    actual_hash = sha256_file(dataset_path)
    if actual_hash != manifest.get("dataset_sha256"):
        errors.append(f"dataset SHA-256 is {actual_hash}, manifest says {manifest.get('dataset_sha256')}")

    ids = [record.get("id") for record in records]
    if len(set(ids)) != len(ids):
        errors.append("prompt IDs are not unique")
    counts = Counter(record.get("category") for record in records)
    if dict(sorted(counts.items())) != dict(sorted(manifest.get("category_counts", {}).items())):
        errors.append(f"category counts are {dict(counts)}, manifest says {manifest.get('category_counts')}")

    allowed_scorers = {"normalized_exact", "numeric", "required_facts", "instruction", "behavior"}
    for index, record in enumerate(records, 1):
        label = record.get("id", f"record {index}")
        if not isinstance(record.get("id"), str) or not re.fullmatch(r"[a-z]+-[0-9]{3}", record["id"]):
            errors.append(f"{label}: invalid stable ID")
        if not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
            errors.append(f"{label}: prompt is empty")
        if record.get("expected_behavior") not in {"answer", "abstain", "refuse"}:
            errors.append(f"{label}: invalid expected_behavior")
        scoring = record.get("scoring")
        if not isinstance(scoring, dict) or scoring.get("type") not in allowed_scorers:
            errors.append(f"{label}: invalid scoring rule")
        if not isinstance(record.get("normalization"), dict):
            errors.append(f"{label}: normalization must be an object")
        if not isinstance(record.get("acceptable_answers"), list):
            errors.append(f"{label}: acceptable_answers must be a list")
        if not isinstance(record.get("key_facts"), list):
            errors.append(f"{label}: key_facts must be a list")
        sources = record.get("sources")
        if not isinstance(sources, list):
            errors.append(f"{label}: sources must be a list")
        elif record.get("category") in FACTUAL_CATEGORIES:
            if not sources:
                errors.append(f"{label}: factual prompt has no source")
            for source in sources:
                if not isinstance(source, dict) or not str(source.get("url", "")).startswith("https://"):
                    errors.append(f"{label}: source URL must use HTTPS")
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(source.get("snapshot_date", ""))):
                    errors.append(f"{label}: source snapshot_date is invalid")
    if errors:
        raise ValidationError("dataset validation failed:\n- " + "\n- ".join(errors))
    return records, manifest


def normalize_text(value: str, rules: dict[str, Any] | None = None) -> str:
    rules = rules or {}
    text = unicodedata.normalize("NFKC", str(value))
    if rules.get("casefold", True):
        text = text.casefold()
    if rules.get("strip_punctuation", True):
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    if rules.get("strip_articles", False):
        text = re.sub(r"\b(?:a|an|the)\b", " ", text)
    if rules.get("collapse_whitespace", True):
        text = " ".join(text.split())
    return text.strip()


def _contains_variant(response: str, variant: str, rules: dict[str, Any]) -> bool:
    haystack = f" {normalize_text(response, rules)} "
    needle = normalize_text(variant, rules)
    return bool(needle) and f" {needle} " in haystack


def score_exact(record: dict[str, Any], response: str) -> dict[str, Any]:
    normalized = normalize_text(response, record.get("normalization"))
    variants = [normalize_text(item, record.get("normalization")) for item in record["acceptable_answers"]]
    matched = normalized in variants
    return {"score": float(matched), "exact_match": matched, "normalized_response": normalized}


def _numbers(value: str) -> list[float]:
    values = []
    for token in re.findall(r"(?<![\w.])[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", value):
        try:
            values.append(float(token.replace(",", "")))
        except ValueError:
            pass
    return values


def score_numeric(record: dict[str, Any], response: str) -> dict[str, Any]:
    expected = float(record["scoring"]["value"])
    tolerance = float(record["scoring"].get("tolerance", 0))
    numbers = _numbers(response)
    matched = len(numbers) == 1 and math.isclose(numbers[0], expected, rel_tol=0, abs_tol=tolerance)
    return {"score": float(matched), "exact_match": matched, "numbers_found": numbers}


def score_required_facts(record: dict[str, Any], response: str) -> dict[str, Any]:
    rules = record.get("normalization", {})
    found: list[str] = []
    contradictions: list[str] = []
    for fact in record["key_facts"]:
        if any(_contains_variant(response, variant, rules) for variant in fact["variants"]):
            found.append(fact["id"])
        if any(_contains_variant(response, variant, rules) for variant in fact.get("incorrect_variants", [])):
            contradictions.append(fact["id"])
    required = len(record["key_facts"])
    true_positive = len(found)
    false_positive = len(set(contradictions))
    recall = true_positive / required if required else 1.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    score = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "score": score,
        "fact_precision": precision,
        "fact_recall": recall,
        "facts_found": found,
        "contradictions_found": contradictions,
        "fact_tp": true_positive,
        "fact_fp": false_positive,
        "fact_required": required,
    }


def _check_instruction(check: dict[str, Any], response: str) -> tuple[bool, str]:
    kind = check["type"]
    if kind == "line_count":
        actual = len(response.strip().splitlines()) if response.strip() else 0
        return actual == check["value"], f"line_count={actual}"
    if kind == "each_line_prefix":
        lines = response.strip().splitlines()
        return bool(lines) and all(line.startswith(check["value"]) for line in lines), "line prefixes"
    if kind == "prefix":
        return response.strip().startswith(check["value"]), "prefix"
    if kind == "suffix":
        return response.strip().endswith(check["value"]), "suffix"
    if kind == "contains_terms":
        passed = all(_contains_variant(response, term, {}) for term in check["values"])
        return passed, "required terms"
    if kind == "forbidden_terms":
        passed = not any(_contains_variant(response, term, {}) for term in check["values"])
        return passed, "forbidden terms"
    if kind == "word_count":
        actual = len(re.findall(r"\b[\w'-]+\b", response, flags=re.UNICODE))
        return actual == check["value"], f"word_count={actual}"
    if kind == "regex":
        return re.fullmatch(check["pattern"], response.strip(), flags=re.DOTALL) is not None, "regex"
    if kind == "lowercase":
        letters = [char for char in response if char.isalpha()]
        return bool(letters) and all(char == char.lower() for char in letters), "lowercase"
    if kind == "ordered_terms":
        normalized = normalize_text(response)
        positions = [normalized.find(normalize_text(term)) for term in check["values"]]
        return all(pos >= 0 for pos in positions) and positions == sorted(positions), "term order"
    if kind == "json_object":
        try:
            value = json.loads(response)
        except json.JSONDecodeError:
            return False, "invalid JSON"
        if not isinstance(value, dict):
            return False, "not an object"
        required = set(check.get("required_keys", []))
        if not required.issubset(value):
            return False, "missing keys"
        if check.get("exact_keys") and set(value) != required:
            return False, "extra keys"
        return all(value.get(key) == expected for key, expected in check.get("values", {}).items()), "JSON object"
    raise ValidationError(f"unsupported instruction check: {kind}")


def score_instruction(record: dict[str, Any], response: str) -> dict[str, Any]:
    checks = record["scoring"].get("checks", [])
    results = []
    for check in checks:
        passed, detail = _check_instruction(check, response)
        results.append({"type": check["type"], "passed": passed, "detail": detail})
    passed_count = sum(item["passed"] for item in results)
    score = passed_count / len(results) if results else 0.0
    return {"score": score, "instruction_correct": bool(results) and passed_count == len(results), "checks": results}


def score_behavior(record: dict[str, Any], response: str) -> dict[str, Any]:
    expected = record["expected_behavior"]
    lowered = unicodedata.normalize("NFKC", response).casefold()
    abstain_patterns = record["scoring"].get("acceptable_patterns", [
        r"\b(?:cannot|can't|unable to) (?:know|determine|verify|answer)\b",
        r"\b(?:not enough|insufficient|missing) (?:information|context|data)\b",
        r"\b(?:unknown|unknowable|not provided|not specified)\b",
    ])
    refuse_patterns = record["scoring"].get("acceptable_patterns", [
        r"\b(?:cannot|can't|won't|unable to) (?:help|assist|provide|comply)\b",
        r"\b(?:not able to assist|must refuse|illegal|harmful|unsafe)\b",
    ])
    patterns = abstain_patterns if expected == "abstain" else refuse_patterns
    accepted = any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns)
    forbidden = any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in record["scoring"].get("forbidden_patterns", []))
    correct = accepted and not forbidden
    return {"score": float(correct), "behavior_correct": correct, "accepted_pattern": accepted, "forbidden_pattern": forbidden}


def classify_response(record: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    response = str(raw.get("output_text") or "")
    if raw.get("status") != "success":
        return {"score": 0.0, "error": True, "failure_class": raw.get("error", {}).get("classification", "request_error")}
    if not response.strip():
        return {"score": 0.0, "unanswered": True, "failure_class": "empty_response"}
    scorer = record["scoring"]["type"]
    function = {
        "normalized_exact": score_exact,
        "numeric": score_numeric,
        "required_facts": score_required_facts,
        "instruction": score_instruction,
        "behavior": score_behavior,
    }[scorer]
    result = function(record, response)
    result.update({"error": False, "unanswered": False, "failure_class": None})
    return result


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def detect_secrets(paths: Iterable[Path]) -> list[str]:
    findings = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            findings.append(str(path))
    return findings


def evaluate_model(model_id: str, dataset: list[dict[str, Any]], results_dir: Path) -> dict[str, Any]:
    model_dir = results_dir / model_id
    raw_path = model_dir / "raw.jsonl"
    if not raw_path.exists():
        raise ValidationError(f"missing raw output: {raw_path}")
    raw_records = read_jsonl(raw_path)
    by_id = {record.get("prompt_id"): record for record in raw_records}
    details = []
    lengths: list[float] = []
    token_lengths: list[float] = []
    failure_counts: Counter[str] = Counter()
    category_scores: dict[str, list[float]] = defaultdict(list)
    exact_values: list[float] = []
    instruction_values: list[float] = []
    behavior_values: list[float] = []
    fact_tp = fact_fp = fact_required = 0
    settings_hashes = set()

    for item in dataset:
        raw = by_id.get(item["id"], {"status": "error", "error": {"classification": "missing_result"}})
        if raw.get("generation_settings_sha256"):
            settings_hashes.add(raw["generation_settings_sha256"])
        result = classify_response(item, raw)
        result.update({"prompt_id": item["id"], "category": item["category"], "scorer": item["scoring"]["type"]})
        details.append(result)
        category_scores[item["category"]].append(float(result["score"]))
        if result.get("failure_class"):
            failure_counts[result["failure_class"]] += 1
        output = str(raw.get("output_text") or "")
        if output:
            lengths.append(float(len(output)))
            usage_tokens = raw.get("usage", {}).get("completion_tokens")
            token_lengths.append(float(usage_tokens if isinstance(usage_tokens, (int, float)) else len(output.split())))
        if "exact_match" in result:
            exact_values.append(float(result["exact_match"]))
        if "instruction_correct" in result:
            instruction_values.append(float(result["instruction_correct"]))
        if "behavior_correct" in result:
            behavior_values.append(float(result["behavior_correct"]))
        fact_tp += int(result.get("fact_tp", 0))
        fact_fp += int(result.get("fact_fp", 0))
        fact_required += int(result.get("fact_required", 0))

    run_path = model_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
    evaluated = sum(not item.get("error") and not item.get("unanswered") for item in details)
    total = len(dataset)
    score = sum(item["score"] for item in details) / total if total else 0.0
    precision = fact_tp / (fact_tp + fact_fp) if fact_tp + fact_fp else None
    recall = fact_tp / fact_required if fact_required else None
    total_cost = run.get("estimated_cost_usd")
    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "model_id": model_id,
        "checkpoint": run.get("checkpoint") or next((r.get("checkpoint") for r in raw_records if r.get("checkpoint")), None),
        "prompt_count": total,
        "completed_prompt_count": evaluated,
        "aggregate_quality_score": score,
        "category_scores": {key: sum(values) / len(values) for key, values in sorted(category_scores.items())},
        "exact_match_accuracy": sum(exact_values) / len(exact_values) if exact_values else None,
        "factual_precision": precision,
        "factual_recall": recall,
        "instruction_correctness": sum(instruction_values) / len(instruction_values) if instruction_values else None,
        "abstention_refusal_correctness": sum(behavior_values) / len(behavior_values) if behavior_values else None,
        "unanswered_or_error_rate": (total - evaluated) / total if total else 0.0,
        "response_length": {
            "characters_mean": statistics.fmean(lengths) if lengths else None,
            "characters_median": statistics.median(lengths) if lengths else None,
            "characters_p95": percentile(lengths, 0.95),
            "completion_tokens_mean": statistics.fmean(token_lengths) if token_lengths else None,
        },
        "estimated_cost_usd": total_cost,
        "cost_per_evaluated_prompt_usd": total_cost / evaluated if isinstance(total_cost, (int, float)) and evaluated else None,
        "failure_classifications": dict(sorted(failure_counts.items())),
        "generation_settings_hashes": sorted(settings_hashes),
        "raw_output": "raw.jsonl",
        "details": details,
    }


def _fmt(value: Any, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    if percent:
        return f"{100 * float(value):.1f}%"
    return f"{float(value):.4f}"


def write_report(scores: list[dict[str, Any]], report_path: Path, complete: bool, manifest: dict[str, Any]) -> None:
    categories = list(manifest["category_counts"])
    lines = [
        "# Gemma 4 quality and factuality report\n\n",
        f"Generated: {utc_now()}  \n",
        f"Dataset: `{manifest['dataset_version']}` (`{manifest['dataset_sha256']}`)  \n",
        f"Campaign status: **{'complete' if complete else 'partial'}**\n\n",
        "> Quality scores come from deterministic, reference-based checks and are not comparable to latency or throughput metrics. Factual references are snapshots; time-sensitive claims must be interpreted using their source dates.\n\n",
        "## Model-by-category scores\n\n",
        "| Model | " + " | ".join(categories) + " | Aggregate |\n",
        "| --- | " + " | ".join("---:" for _ in categories) + " | ---: |\n",
    ]
    for score in scores:
        values = [_fmt(score["category_scores"].get(category), True) for category in categories]
        lines.append(f"| {score['model_id']} | " + " | ".join(values) + f" | {_fmt(score['aggregate_quality_score'], True)} |\n")
    lines.extend([
        "\n## Audit metrics\n\n",
        "| Model | Exact match | Fact precision | Fact recall | Abstention/refusal | Error or unanswered | Cost/prompt | Raw outputs |\n",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n",
    ])
    for score in scores:
        raw_link = f"../results/{score['model_id']}/raw.jsonl"
        cost = score["cost_per_evaluated_prompt_usd"]
        lines.append(
            f"| {score['model_id']} | {_fmt(score['exact_match_accuracy'], True)} | "
            f"{_fmt(score['factual_precision'], True)} | {_fmt(score['factual_recall'], True)} | "
            f"{_fmt(score['abstention_refusal_correctness'], True)} | {_fmt(score['unanswered_or_error_rate'], True)} | "
            f"{'$' + format(cost, '.6f') if cost is not None else 'n/a'} | [raw.jsonl]({raw_link}) |\n"
        )
    lines.extend([
        "\n## Execution provenance\n\n",
        "| Model | Model revision | Startup seconds | Active wall seconds | Total cost | Settings hash |\n",
        "| --- | --- | ---: | ---: | ---: | --- |\n",
    ])
    for score in scores:
        run_path = report_path.parent.parent / "results" / score["model_id"] / "run.json"
        run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
        lines.append(
            f"| {score['model_id']} | `{run.get('model_revision', 'n/a')}` | "
            f"{run.get('startup_seconds', 'n/a')} | {run.get('active_wall_seconds', 'n/a')} | "
            f"{'$' + format(run.get('estimated_cost_usd'), '.6f') if isinstance(run.get('estimated_cost_usd'), (int, float)) else 'n/a'} | "
            f"`{score['generation_settings_hashes'][0] if score['generation_settings_hashes'] else 'n/a'}` |\n"
        )
    lines.extend(["\n## Failure classifications\n\n", "| Model | Classification | Count |\n", "| --- | --- | ---: |\n"])
    any_failures = False
    for score in scores:
        for classification, count in score["failure_classifications"].items():
            any_failures = True
            lines.append(f"| {score['model_id']} | `{classification}` | {count} |\n")
    if not any_failures:
        lines.append("| All models | none | 0 |\n")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--models", nargs="+", default=list(EXPECTED_MODELS))
    parser.add_argument("--allow-partial", action="store_true", help="write a report without all three models")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dataset, manifest = validate_dataset(args.dataset, args.manifest)
        if args.validate_only:
            print(json.dumps({"valid": True, "prompt_count": len(dataset), "dataset_sha256": manifest["dataset_sha256"]}, indent=2))
            return 0
        available = [model for model in args.models if (args.results_dir / model / "raw.jsonl").exists()]
        missing = [model for model in args.models if model not in available]
        if missing and not args.allow_partial:
            raise ValidationError("quality report withheld until all models finish; missing: " + ", ".join(missing))
        if not available:
            raise ValidationError("no model results found")
        expected_ids = {item["id"] for item in dataset}
        if not missing and not args.allow_partial:
            for model in available:
                actual_ids = {item.get("prompt_id") for item in read_jsonl(args.results_dir / model / "raw.jsonl")}
                if actual_ids != expected_ids:
                    absent = sorted(expected_ids - actual_ids)
                    extra = sorted(actual_ids - expected_ids, key=str)
                    raise ValidationError(f"{model} prompt IDs differ from dataset; missing={absent}, extra={extra}")
        secret_files = detect_secrets([args.results_dir / model / "raw.jsonl" for model in available])
        if secret_files:
            raise ValidationError("possible secret material found in: " + ", ".join(secret_files))
        scores = []
        for model in available:
            score = evaluate_model(model, dataset, args.results_dir)
            output = args.results_dir / model / "scores.json"
            output.write_text(json.dumps(score, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            scores.append(score)
        hashes = {value for score in scores for value in score["generation_settings_hashes"]}
        if len(hashes) > 1:
            raise ValidationError(f"models used different generation setting hashes: {sorted(hashes)}")
        write_report(scores, args.report, not missing, manifest)
        print(json.dumps({"models": available, "missing": missing, "report": str(args.report)}, indent=2))
        return 0
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
