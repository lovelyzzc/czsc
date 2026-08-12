"""Audit the immutable XS-Chan Stage 3 chain after historical-input cleanup.

The frozen Stage 3 source cannot be edited without invalidating its July 2026
anchor.  This external auditor therefore verifies the untouched source/spec,
the append-only ledger, workspace bytes, and recoverable Git blobs separately.
It never upgrades Stage 3 to confirmation or live authority.

Run: ``uv run --no-sync python scripts/xs_chan_stage3_repro_audit.py``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import xs_chan_exploration_stage3 as stage3

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "_output" / "xs_chan_stage3_repro_audit"
AUDIT_PATH = OUTPUT_DIR / "audit.json"
SCHEMA = "xs_chan_stage3_repro_audit_v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_blob(commit: str, relative_path: str) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 else None


def git_blob_record(commit: str, relative_path: str, expected_sha256: str) -> dict[str, Any]:
    """Report whether an exact frozen tracked input is recoverable from Git."""

    raw = _git_blob(commit, relative_path)
    actual = _sha256_bytes(raw) if raw is not None else None
    return {
        "commit": commit,
        "path": relative_path,
        "available": raw is not None,
        "sha256": actual,
        "expected_sha256": expected_sha256,
        "matches": actual == expected_sha256,
    }


def workspace_record(relative_path: str, expected_sha256: str) -> dict[str, Any]:
    """Report exact current-workspace availability without substituting a Git blob."""

    path = REPO_ROOT / relative_path
    actual = stage3.sha256_file(path) if path.is_file() else None
    return {
        "path": relative_path,
        "available": path.is_file(),
        "sha256": actual,
        "expected_sha256": expected_sha256,
        "matches": actual == expected_sha256,
    }


def _stage2_parent_evidence(spec: Mapping[str, Any], parent_commit: str) -> dict[str, Any]:
    parent = spec["frozen_stage2_parent"]
    declarations = {
        "spec": (str(parent["spec_path"]), str(parent["spec_physical_sha256"])),
        "source": (str(parent["source_path"]), str(parent["source_sha256"])),
        "results": (str(parent["results_path"]), str(parent["results_sha256"])),
    }
    return {
        label: {
            "workspace": workspace_record(path, expected),
            "git_parent": git_blob_record(parent_commit, path, expected),
        }
        for label, (path, expected) in declarations.items()
    }


def _stage1_evidence(spec: Mapping[str, Any], parent_commit: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for declaration in spec["frozen_stage1_path_bootstrap"]:
        name = str(declaration["name"])
        path = str(declaration["path"])
        expected = str(declaration["sha256"])
        result[name] = {
            "workspace": workspace_record(path, expected),
            "git_parent": git_blob_record(parent_commit, path, expected),
        }
    return result


def _anchor_evidence(spec: Mapping[str, Any]) -> dict[str, Any]:
    anchor = stage3.read_json(stage3.ANCHOR_PATH)
    expected = {
        "schema": stage3.ANCHOR_SCHEMA,
        "study_id": spec["study_id"],
        "study_identity": stage3.study_identity(spec),
        "spec_path": str(stage3.SPEC_PATH.relative_to(stage3.REPO_ROOT)),
        "spec_physical_sha256": stage3.sha256_file(stage3.SPEC_PATH),
        "spec_canonical_sha256": stage3.sha256_bytes(stage3.canonical_json(spec)),
        "source_path": str(stage3.SOURCE_PATH.relative_to(stage3.REPO_ROOT)),
        "source_sha256": stage3.sha256_file(stage3.SOURCE_PATH),
        "frozen_at_utc": spec["frozen_at_utc"],
        "first_prospective_decision_date": spec["historical_exclusion"]["first_prospective_decision_date"],
        "parent_git_commit": spec["freeze_evidence"]["parent_git_commit"],
        "external_timestamp_or_signature": False,
        "validity_requirement": spec["freeze_evidence"]["required_before_first_decision"],
    }
    pushed_commit: str | None = None
    pushed_error: str | None = None
    try:
        pushed_commit = stage3.validate_protocol_anchor(spec)
    except Exception as exc:  # Evidence report must preserve rather than hide anchor failures.
        pushed_error = f"{type(exc).__name__}: {exc}"
    return {
        "bytes_match_frozen_source_and_spec": anchor == expected,
        "source_sha256": expected["source_sha256"],
        "spec_physical_sha256": expected["spec_physical_sha256"],
        "study_identity": expected["study_identity"],
        "pushed_anchor_commit": pushed_commit,
        "pushed_anchor_error": pushed_error,
        "external_timestamp_or_signature": False,
    }


def _ledger_evidence(spec: Mapping[str, Any]) -> dict[str, Any]:
    root = stage3.resolve_ledger_root(spec)
    records = stage3.scan_records(root)
    status = stage3.status_report(spec, records, root=root)
    counts: dict[str, int] = {}
    for record in records:
        record_type = str(record.data["record_type"])
        counts[record_type] = counts.get(record_type, 0) + 1
    head = records[-1].data if records else None
    anchor_path = (
        SCRIPT_DIR / "xs_chan_exploration_stage3_ledger_anchors" / f"{head['sequence']:06d}_{head['record_hash']}.json"
        if head is not None
        else None
    )
    tracked_head = stage3.read_json(anchor_path) if anchor_path is not None and anchor_path.is_file() else None
    return {
        "root": str(root.relative_to(REPO_ROOT)),
        "records": len(records),
        "record_types": counts,
        "head_sequence": int(head["sequence"]) if head is not None else None,
        "head_hash": str(head["record_hash"]) if head is not None else None,
        "tracked_head_anchor_available": tracked_head is not None,
        "tracked_head_anchor_matches": bool(
            tracked_head is not None
            and head is not None
            and tracked_head.get("record_hash") == head["record_hash"]
            and tracked_head.get("study_identity") == stage3.study_identity(spec)
        ),
        "blinded_status": status,
    }


def run_audit() -> dict[str, Any]:
    """Reconcile the immutable chain, Git history, and missing local data."""

    spec = stage3.read_json(stage3.SPEC_PATH)
    parent_commit = str(spec["freeze_evidence"]["parent_git_commit"])
    anchor = _anchor_evidence(spec)
    stage2 = _stage2_parent_evidence(spec, parent_commit)
    stage1 = _stage1_evidence(spec, parent_commit)
    ledger = _ledger_evidence(spec)
    current_loader_error: str | None = None
    try:
        stage3.load_and_validate_spec()
    except Exception as exc:
        current_loader_error = f"{type(exc).__name__}: {exc}"

    tracked_parent_recoverable = all(item["git_parent"]["matches"] for item in stage2.values()) and all(
        item["git_parent"]["matches"] for item in stage1.values() if item["git_parent"]["available"]
    )
    workspace_complete = all(item["workspace"]["matches"] for item in stage2.values()) and all(
        item["workspace"]["matches"] for item in stage1.values()
    )
    prospective_decisions = int(ledger["record_types"].get("decision_freeze", 0))
    protocol_ledger_valid = bool(
        anchor["bytes_match_frozen_source_and_spec"]
        and ledger["records"] >= 1
        and ledger["tracked_head_anchor_matches"]
    )
    if not protocol_ledger_valid:
        status = "STAGE3_PROTOCOL_OR_LEDGER_INVALID"
    elif prospective_decisions == 0:
        status = "STAGE3_PROTOCOL_LEDGER_VALID_NO_PROSPECTIVE_DECISIONS"
    elif not workspace_complete:
        status = "STAGE3_LEDGER_VALID_PARENT_WORKSPACE_INCOMPLETE"
    else:
        status = "STAGE3_FORWARD_LEDGER_IN_PROGRESS"
    audit = {
        "schema": SCHEMA,
        "study_id": spec["study_id"],
        "anchor": anchor,
        "frozen_stage2_parent": stage2,
        "frozen_stage1_bootstrap": stage1,
        "ledger": ledger,
        "reproducibility": {
            "tracked_parent_git_blobs_recoverable": tracked_parent_recoverable,
            "workspace_exactly_rehydrated": workspace_complete,
            "original_loader_error": current_loader_error,
            "missing_untracked_frozen_inputs": [
                name
                for name, item in stage1.items()
                if not item["workspace"]["matches"] and not item["git_parent"]["available"]
            ],
        },
        "verdict": {
            "status": status,
            "confirmation_chain": "NOT_STARTED",
            "prospective_decisions": prospective_decisions,
            "prospective_labels": int(ledger["record_types"].get("label_completion", 0)),
            "efficacy_evaluation_permitted": False,
            "live_authorized": False,
        },
        "limitations": [
            "The immutable Stage 3 source cannot be patched without invalidating its tracked source anchor.",
            "Historical Stage 1 parquet bootstraps were ignored artifacts and are absent from both workspace and Git.",
            "The Stage 2 source/spec evolved after the Stage 3 parent freeze; current paths cannot equal both revisions.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    audit = run_audit()
    print(json.dumps(audit["verdict"], ensure_ascii=False, indent=2))
    print(f"[output] {AUDIT_PATH}")


if __name__ == "__main__":
    main()
