#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


DEFAULT_MODEL = os.getenv("LOCAL_QWEN_MODEL", "Qwen/Qwen3-8B-AWQ")
DEFAULT_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1/chat/completions")
DEFAULT_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "EMPTY")


def main() -> int:
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": "Reply with JSON only."},
            {"role": "user", "content": "Return {\"status\": \"ok\", \"backend\": \"local-qwen\"} and nothing else.\n/no_think"},
        ],
        "temperature": 0.0,
        "max_tokens": 80,
        "stream": False,
    }
    request = urllib.request.Request(
        url=DEFAULT_BASE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEFAULT_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise SystemExit(f"Local server is not reachable: {exc}") from exc
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Local server returned HTTP {exc.code}: {detail}") from exc

    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
