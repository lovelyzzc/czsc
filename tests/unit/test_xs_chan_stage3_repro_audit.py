"""Ratchets for the immutable Stage 3 external reproducibility audit."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import xs_chan_stage3_repro_audit as audit  # noqa: E402


def test_frozen_stage2_source_is_recoverable_from_parent_git_object() -> None:
    spec = audit.stage3.read_json(audit.stage3.SPEC_PATH)
    parent = spec["frozen_stage2_parent"]
    result = audit.git_blob_record(
        spec["freeze_evidence"]["parent_git_commit"],
        parent["source_path"],
        parent["source_sha256"],
    )
    assert result["available"] is True
    assert result["matches"] is True


def test_current_stage2_revision_is_not_misrepresented_as_frozen_parent() -> None:
    spec = audit.stage3.read_json(audit.stage3.SPEC_PATH)
    parent = spec["frozen_stage2_parent"]
    result = audit.workspace_record(parent["source_path"], parent["source_sha256"])
    assert result["available"] is True
    assert result["matches"] is False


def test_repro_audit_validates_anchor_and_genesis_without_efficacy() -> None:
    result = audit.run_audit()
    assert result["anchor"]["bytes_match_frozen_source_and_spec"] is True
    assert result["anchor"]["pushed_anchor_error"] is None
    assert result["ledger"]["records"] == 1
    assert result["ledger"]["record_types"] == {"genesis": 1}
    assert result["verdict"] == {
        "status": "STAGE3_PROTOCOL_LEDGER_VALID_NO_PROSPECTIVE_DECISIONS",
        "confirmation_chain": "NOT_STARTED",
        "prospective_decisions": 0,
        "prospective_labels": 0,
        "efficacy_evaluation_permitted": False,
        "live_authorized": False,
    }


def test_absent_ignored_stage1_bootstraps_remain_explicit() -> None:
    result = audit.run_audit()
    assert result["reproducibility"]["missing_untracked_frozen_inputs"] == [
        "historical_ranked_surface",
        "historical_buffered_memberships",
        "historical_gate_proposals",
    ]
    assert result["reproducibility"]["workspace_exactly_rehydrated"] is False
