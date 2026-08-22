#!/usr/bin/env python3
"""Publish a scoreboard-rwkv campaign and its task payloads.

The scoreboard API deliberately accepts the versioned publication contracts
defined by scoreboard-rwkv.  This client transports those payloads; it does
not turn an lm-eval ``results_*.json`` summary into a different contract or
invent missing per-sample execution evidence.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, TextIO
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen


CAMPAIGN_SCHEMA = "lighteval-campaign-v3"
TASK_SCHEMA = "lighteval-task-v2"
LIGHTEVAL_VERSION = "0.13.0"


class ScoreboardError(RuntimeError):
    """An actionable local or remote publication error."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ScoreboardError(
            f"cannot read strict JSON object {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ScoreboardError(f"JSON document must be an object: {path}")
    return value


def canonical_json(value: Any) -> bytes:
    """Return the canonical JSON bytes used by scoreboard-rwkv digests."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ScoreboardError(
            f"payload cannot be canonicalized as JSON: {error}"
        ) from error


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _api_root(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ScoreboardError(
            "--base-url must be an absolute http(s) URL, for example "
            "https://eval.rwkv.rs or https://eval.rwkv.rs/test"
        )
    if parsed.query or parsed.fragment:
        raise ScoreboardError("--base-url must not contain a query or fragment")
    path = parsed.path.rstrip("/")
    if not path.endswith("/api"):
        path += "/api"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _response_json(body: bytes, *, url: str) -> dict[str, Any]:
    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise ScoreboardError(
            f"scoreboard returned invalid JSON from {url}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ScoreboardError(f"scoreboard returned a non-object JSON value from {url}")
    return value


def _remote_error(method: str, url: str, status: int, body: bytes) -> ScoreboardError:
    try:
        detail = _response_json(body, url=url)
        rendered = json.dumps(detail, ensure_ascii=False, separators=(",", ":"))
    except ScoreboardError:
        rendered = body.decode("utf-8", errors="replace")
    if len(rendered) > 2000:
        rendered = rendered[:2000] + "..."
    return ScoreboardError(
        f"scoreboard API {method} {url} returned HTTP {status}: {rendered}"
    )


class ScoreboardClient:
    def __init__(self, *, base_url: str, token: str, timeout: float = 3600.0) -> None:
        if not token:
            raise ScoreboardError("a publication token is required")
        if timeout <= 0:
            raise ScoreboardError("--timeout must be positive")
        self.api_root = _api_root(base_url)
        self.token = token
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        url = f"{self.api_root}{path}"
        headers = {"Authorization": f"Bearer {self.token}"}
        body: bytes | None = None
        if payload is not None:
            body = gzip.compress(canonical_json(payload))
            headers.update(
                {
                    "Content-Type": "application/json",
                    "Content-Encoding": "gzip",
                }
            )
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        request = Request(url, data=body, headers=headers, method=method)  # noqa: S310
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                response_body = response.read()
                status = getattr(response, "status", 200)
        except HTTPError as error:
            raise _remote_error(method, url, error.code, error.read()) from error
        except URLError as error:
            raise ScoreboardError(
                f"cannot reach scoreboard API {method} {url}: {error.reason}"
            ) from error
        except OSError as error:
            raise ScoreboardError(
                f"cannot reach scoreboard API {method} {url}: {error}"
            ) from error
        if status < 200 or status >= 300:
            raise _remote_error(method, url, status, response_body)
        return _response_json(response_body, url=url)

    def preflight(self) -> dict[str, Any]:
        response = self._request("GET", "/v1/evaluation-publication-preflight")
        if response.get("status") != "ready":
            raise ScoreboardError(f"scoreboard preflight is not ready: {response}")
        if response.get("schema_version") != CAMPAIGN_SCHEMA:
            raise ScoreboardError(
                "scoreboard schema mismatch: "
                f"expected {CAMPAIGN_SCHEMA}, got {response.get('schema_version')!r}"
            )
        if response.get("lighteval_version") != LIGHTEVAL_VERSION:
            raise ScoreboardError(
                "scoreboard LightEval version mismatch: "
                f"expected {LIGHTEVAL_VERSION}, got {response.get('lighteval_version')!r}"
            )
        return response

    def create_campaign(self, campaign: dict[str, Any]) -> dict[str, Any]:
        run_key = campaign.get("run_key")
        if not isinstance(run_key, str):
            raise ScoreboardError("campaign.run_key must be a string")
        return self._request(
            "POST",
            "/v1/evaluation-campaigns",
            payload=campaign,
            idempotency_key=f"campaign:{run_key}",
        )

    def campaign_status(self, campaign_id: str) -> dict[str, Any]:
        return self._request(
            "GET", f"/v1/evaluation-campaigns/{quote(campaign_id, safe='')}"
        )

    def publish_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        campaign_id = payload.get("campaign_id")
        task = payload.get("task")
        if not isinstance(campaign_id, str) or not isinstance(task, dict):
            raise ScoreboardError(
                "task payload must contain campaign_id and task object"
            )
        identity = task.get("identity")
        if not isinstance(identity, str) or not identity:
            raise ScoreboardError(
                "task payload task.identity must be a non-empty string"
            )
        digest = content_digest(payload)
        return self._request(
            "PUT",
            f"/v1/evaluation-campaigns/{quote(campaign_id, safe='')}/tasks/"
            f"{quote(identity, safe='')}",
            payload=payload,
            idempotency_key=f"publish:{digest}",
        )

    def finalize_campaign(self, campaign_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/evaluation-campaigns/{quote(campaign_id, safe='')}/finalize",
            idempotency_key=f"finalize:{campaign_id}",
        )


def _validate_campaign(campaign: dict[str, Any]) -> list[str]:
    if campaign.get("schema_version") != CAMPAIGN_SCHEMA:
        raise ScoreboardError(
            f"campaign.schema_version must be {CAMPAIGN_SCHEMA!r}; "
            "raw lm-eval results are not scoreboard publication payloads"
        )
    run_key = campaign.get("run_key")
    if not isinstance(run_key, str) or re.fullmatch(r"[0-9a-f]{64}", run_key) is None:
        raise ScoreboardError(
            "campaign.run_key must be a 64-character hexadecimal string"
        )
    if campaign.get("lighteval_version") != LIGHTEVAL_VERSION:
        raise ScoreboardError(
            f"campaign.lighteval_version must be {LIGHTEVAL_VERSION!r}"
        )
    expected = campaign.get("expected_tasks")
    if not isinstance(expected, list) or not expected:
        raise ScoreboardError("campaign.expected_tasks must be a non-empty array")
    identities: list[str] = []
    for index, task in enumerate(expected):
        if not isinstance(task, dict) or not isinstance(task.get("identity"), str):
            raise ScoreboardError(
                f"campaign.expected_tasks[{index}] lacks task identity"
            )
        identity = task["identity"]
        if not identity:
            raise ScoreboardError(
                f"campaign.expected_tasks[{index}] has empty identity"
            )
        identities.append(identity)
    if len(identities) != len(set(identities)):
        raise ScoreboardError("campaign.expected_tasks identities must be unique")
    return identities


def _validate_tasks(
    payloads: list[dict[str, Any]], expected_identities: list[str]
) -> dict[str, dict[str, Any]]:
    by_identity: dict[str, dict[str, Any]] = {}
    for index, payload in enumerate(payloads):
        if payload.get("schema_version") != TASK_SCHEMA:
            raise ScoreboardError(
                f"task file {index} schema_version must be {TASK_SCHEMA!r}"
            )
        task = payload.get("task")
        if not isinstance(task, dict) or not isinstance(task.get("identity"), str):
            raise ScoreboardError(f"task file {index} lacks task.identity")
        identity = task["identity"]
        if not identity:
            raise ScoreboardError(f"task file {index} has empty task.identity")
        if identity in by_identity:
            raise ScoreboardError(f"duplicate task payload for identity {identity}")
        by_identity[identity] = payload
    expected = set(expected_identities)
    actual = set(by_identity)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing=" + ",".join(sorted(missing)))
        if extra:
            parts.append("unexpected=" + ",".join(sorted(extra)))
        raise ScoreboardError(
            "task payload set does not match campaign: " + "; ".join(parts)
        )
    return by_identity


def load_publication_inputs(
    campaign_path: Path, task_paths: list[Path]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[str]]:
    campaign = _load_json_object(campaign_path)
    expected_identities = _validate_campaign(campaign)
    payloads = [_load_json_object(path) for path in task_paths]
    tasks = _validate_tasks(payloads, expected_identities)
    return campaign, tasks, expected_identities


def publish(
    *,
    client: ScoreboardClient,
    campaign: dict[str, Any],
    task_by_identity: dict[str, dict[str, Any]],
    expected_identities: list[str],
    finalize: bool = True,
) -> dict[str, Any]:
    preflight = client.preflight()
    campaign_receipt = client.create_campaign(campaign)
    campaign_id = campaign_receipt.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ScoreboardError(
            f"campaign creation returned no campaign_id: {campaign_receipt}"
        )

    status = client.campaign_status(campaign_id)
    if status.get("campaign_id") != campaign_id:
        raise ScoreboardError("campaign status returned a different campaign_id")
    acknowledged = status.get("acknowledged_task_digests", {})
    if not isinstance(acknowledged, dict):
        raise ScoreboardError("campaign status has invalid acknowledged_task_digests")
    if status.get("status") == "complete":
        mismatched = []
        for identity in expected_identities:
            payload = deepcopy(task_by_identity[identity])
            payload["campaign_id"] = campaign_id
            if acknowledged.get(identity) != content_digest(payload):
                mismatched.append(identity)
        if mismatched:
            raise ScoreboardError(
                "campaign is already complete and has different task content: "
                + ", ".join(mismatched)
            )
        return {
            "campaign_id": campaign_id,
            "campaign": campaign_receipt,
            "preflight": preflight,
            "tasks": [
                {"identity": identity, "disposition": "unchanged"}
                for identity in expected_identities
            ],
            "finalize": {"status": "complete", "task_count": len(expected_identities)},
        }

    task_receipts: list[dict[str, Any]] = []
    for identity in expected_identities:
        payload = deepcopy(task_by_identity[identity])
        payload["campaign_id"] = campaign_id
        digest = content_digest(payload)
        if acknowledged.get(identity) == digest:
            task_receipts.append(
                {
                    "identity": identity,
                    "disposition": "unchanged",
                    "content_digest": digest,
                }
            )
            continue
        receipt = client.publish_task(payload)
        task_receipts.append({"identity": identity, **receipt})

    final_receipt: dict[str, Any] | None = None
    if finalize:
        final_receipt = client.finalize_campaign(campaign_id)
    return {
        "campaign_id": campaign_id,
        "campaign": campaign_receipt,
        "preflight": preflight,
        "tasks": task_receipts,
        "finalize": final_receipt,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload scoreboard-rwkv campaign/task publication JSON payloads."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SCOREBOARD_BASE_URL"),
        help="scoreboard origin or deployment prefix (env: SCOREBOARD_BASE_URL)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("SCOREBOARD_PUBLICATION_TOKEN"),
        help="publication token (env: SCOREBOARD_PUBLICATION_TOKEN)",
    )
    parser.add_argument("--campaign", type=Path, help="lighteval-campaign-v3 JSON file")
    parser.add_argument(
        "--task",
        type=Path,
        action="append",
        default=[],
        help="lighteval-task-v2 JSON file; repeat once per expected task",
    )
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate local payloads without contacting the scoreboard",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="only verify the remote publication contract and token",
    )
    parser.add_argument(
        "--no-finalize",
        action="store_true",
        help="upload tasks but leave the campaign incomplete",
    )
    return parser


def _require_credentials(args: argparse.Namespace) -> tuple[str, str]:
    if not args.base_url:
        raise ScoreboardError("--base-url or SCOREBOARD_BASE_URL is required")
    if not args.token:
        raise ScoreboardError(
            "--token or SCOREBOARD_PUBLICATION_TOKEN is required; never put the token in a file"
        )
    return args.base_url, args.token


def _write_json(value: dict[str, Any], output: TextIO | None = None) -> None:
    if output is None:
        output = sys.stdout
    json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
    output.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.preflight_only and args.dry_run:
            raise ScoreboardError("--preflight-only and --dry-run cannot be combined")
        if args.preflight_only:
            base_url, token = _require_credentials(args)
            client = ScoreboardClient(
                base_url=base_url, token=token, timeout=args.timeout
            )
            _write_json(client.preflight())
            return 0
        if args.campaign is None or not args.task:
            raise ScoreboardError("--campaign and at least one --task are required")
        campaign, task_by_identity, expected_identities = load_publication_inputs(
            args.campaign, args.task
        )
        if args.dry_run:
            _write_json(
                {
                    "dry_run": True,
                    "campaign_run_key": campaign["run_key"],
                    "expected_task_count": len(expected_identities),
                    "task_identities": expected_identities,
                }
            )
            return 0
        base_url, token = _require_credentials(args)
        client = ScoreboardClient(base_url=base_url, token=token, timeout=args.timeout)
        result = publish(
            client=client,
            campaign=campaign,
            task_by_identity=task_by_identity,
            expected_identities=expected_identities,
            finalize=not args.no_finalize,
        )
        _write_json(result)
        return 0
    except (ScoreboardError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
