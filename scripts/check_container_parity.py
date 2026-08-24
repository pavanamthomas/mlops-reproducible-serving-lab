"""Compare a running container's probability with an offline fitted bundle.

This is an end-to-end contract check across the HTTP boundary.  It is not a
latency benchmark, uptime test, or claim of production deployment.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

from mlserv.api.serving import frame_from_records, predict_proba_1
from mlserv.artifacts import load_bundle

FIXTURE: dict[str, float | str] = {
    "age": 45.0,
    "income": 80000.0,
    "credit_score": 0.72,
    "segment": "B",
}


def _json_request(url: str, *, payload: dict[str, object] | None = None) -> dict[str, object]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=3.0) as response:  # noqa: S310 - fixed local CI URL
        return json.loads(response.read().decode("utf-8"))


def wait_for_health(base_url: str, *, attempts: int = 30, delay: float = 1.0) -> dict[str, object]:
    """Poll until the API is ready, treating startup socket resets as transient."""
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            health = _json_request(f"{base_url.rstrip('/')}/health")
            if health.get("status") == "ok" and health.get("model_loaded") is True:
                return health
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            json.JSONDecodeError,
        ) as exc:
            # A container can accept a TCP connection before uvicorn has fully
            # initialised the application. A reset at that boundary is a retry
            # condition, not evidence that prediction parity failed.
            last_error = exc
        time.sleep(delay)
    raise RuntimeError(f"container did not become healthy: {last_error}")


def check_parity(artifact: Path, base_url: str, *, tolerance: float) -> tuple[float, float]:
    bundle = load_bundle(artifact)
    frame = frame_from_records([FIXTURE])
    offline = float(predict_proba_1(bundle.pipeline, frame)[0])

    wait_for_health(base_url)
    response = _json_request(f"{base_url.rstrip('/')}/predict-proba", payload=FIXTURE)
    served = float(response["probability_1"])

    if response.get("model_version") != bundle.model_version:
        raise AssertionError(
            f"model version mismatch: served={response.get('model_version')} offline={bundle.model_version}"
        )
    if response.get("schema_version") != bundle.schema_version:
        raise AssertionError(
            f"schema version mismatch: served={response.get('schema_version')} offline={bundle.schema_version}"
        )
    if not math.isclose(served, offline, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(
            f"prediction parity failed: served={served:.17g}, offline={offline:.17g}, tolerance={tolerance:g}"
        )
    return offline, served


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()

    offline, served = check_parity(args.artifact, args.base_url, tolerance=args.tolerance)
    print(json.dumps({"offline_probability": offline, "served_probability": served, "abs_error": abs(served - offline)}))


if __name__ == "__main__":
    main()
