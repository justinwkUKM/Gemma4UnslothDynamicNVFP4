#!/usr/bin/env python3
"""Stream plain-text Gemma responses from a local vLLM OpenAI endpoint."""

from __future__ import annotations

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream a Gemma response through the local SSH tunnel."
    )
    parser.add_argument("prompt", help="Prompt text to send to the model")
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000",
        help="vLLM base URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--model",
        default="gemma-4-26B-A4B-it-NVFP4",
        help="Served model name",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
    }
    request = Request(
        f"{args.endpoint.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=600) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    content = chunk["choices"][0].get("delta", {}).get("content")
                except (KeyError, IndexError, json.JSONDecodeError):
                    continue
                if content:
                    print(content, end="", flush=True)
            print()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {detail}", file=sys.stderr)
        return 1
    except URLError as exc:
        print(
            f"Could not connect to {args.endpoint}. Is the SSH tunnel running? {exc.reason}",
            file=sys.stderr,
        )
        return 1
    except TimeoutError:
        print("Request timed out while waiting for the model.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
