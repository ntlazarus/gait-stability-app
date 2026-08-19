"""Deterministic tests for Step 4b review resolution."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from gait_stability.gait_events import (
    GAIT_EVENT_FIELDS,
    STRIDE_FIELDS,
    GaitEvent,
    GaitEventConfig,
    Stride,
)
from gait_stability.review_resolution import (
    INPUT_ARTIFACT_NAMES,
    OUTPUT_ARTIFACT_NAMES,
    REVIEW_RESOLUTION_ALGORITHM_VERSION,
    REVIEW_RESOLUTION_SCHEMA_VERSION,
    ReviewResolutionArtifacts,
    ReviewResolutionArtifactValidationError,
    ReviewResolutionError,
    resolve_gait_reviews,
)

# ---------------------------------------------------------------------------
# Irregular canonical timestamps (not uniform spacing)
# ---------------------------------------------------------------------------
_FRAME_TS: dict[int, float] = {
    10: 0.333,
    20: 0.667,
    30: 1.100,
    40: 1.533,
    50: 1.967,
    60: 2.400,
    70: 2.833,
    80: 3.267,
    82: 3.367,
    86: 3.500,
    90: 3.700,
    100: 4.133,
    110: 4.567,
    120: 5.000,
    130: 5.433,
    138: 5.700,
    140: 5.800,
    150: 6.233,
    160: 6.667,
}


# ---------------------------------------------------------------------------
# Minimal GaitEvent / Stride constructors
# ---------------------------------------------------------------------------


def _make_event(
    event_id: str,
    frame: int,
    side: str,
    status: str = "accepted",
    included: bool = True,
    quality: str = "high",
    **overrides: Any,
) -> GaitEvent:
    ts = _FRAME_TS[frame]
    defaults: dict[str, Any] = {
        "event_id": event_id,
        "frame_index": frame,
        "timestamp_seconds": ts,
        "side": side,
        "event_type": "candidate_initial_contact",
        "detection_method": "test",
        "detection_status": status,
        "included_in_stride_construction": included,
        "confidence_or_quality": quality,
        "peak_value": 0.15,
        "prominence": 0.05,
        "pre_velocity": 0.1,
        "post_velocity": -0.1,
        "plateau_start_frame": frame,
        "plateau_end_frame": frame,
        "raw_peak_frame": frame,
        "raw_peak_offset_frames": 0,
        "ankle_peak_frame": frame,
        "ankle_peak_offset_frames": 0,
        "ankle_support_observed_usable": True,
        "ankle_support_interpolated": False,
        "ankle_support_smoothing_contains_interpolation": False,
        "primary_support_observed_usable": True,
        "primary_support_interpolated": False,
        "primary_support_smoothing_contains_interpolation": False,
        "signal_support_notes": (),
        "sequence_context_notes": (),
        "source": "automatic",
        "review_status": "unreviewed",
        "rejection_reasons": (),
    }
    defaults.update(overrides)
    return GaitEvent(**defaults)


def _make_stride(
    stride_id: str,
    side: str,
    start_id: str,
    end_id: str,
    start_frame: int,
    end_frame: int,
    **overrides: Any,
) -> Stride:
    defaults: dict[str, Any] = {
        "stride_id": stride_id,
        "side": side,
        "start_event_id": start_id,
        "end_event_id": end_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_timestamp_seconds": _FRAME_TS[start_frame],
        "end_timestamp_seconds": _FRAME_TS[end_frame],
        "duration_seconds": _FRAME_TS[end_frame] - _FRAME_TS[start_frame],
        "quality": "high",
        "contralateral_event_id": None,
        "contralateral_event_count": 0,
        "sequence_notes": (),
        "source": "automatic",
        "review_status": "unreviewed",
    }
    defaults.update(overrides)
    return Stride(**defaults)


# ---------------------------------------------------------------------------
# Synthetic artifact directory builder
# ---------------------------------------------------------------------------


def _write_csv(
    path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_val(v) for k, v in row.items()})


def _csv_val(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return "|".join(str(x) for x in v)
    return str(v)


def _event_row(e: GaitEvent) -> dict[str, Any]:
    return {f: getattr(e, f) for f in GAIT_EVENT_FIELDS}


def _stride_row(s: Stride) -> dict[str, Any]:
    return {f: getattr(s, f) for f in STRIDE_FIELDS}


def _build_artifact_dir(
    tmp_path: Path,
    *,
    events: list[GaitEvent],
    strides: list[Stride],
    event_reviews: list[dict[str, str]],
    stride_reviews: list[dict[str, str]],
    preprocessing_inputs: dict[str, Any] | None = None,
    include_pose_frames: bool = True,
    include_preprocessing: bool = True,
) -> Path:
    """Write all 7 artifact files to tmp_path and return the directory."""
    # pose_frames.csv
    if include_pose_frames:
        rows = [
            {
                "frame_index": fi,
                "nominal_timestamp_seconds": ts,
                "backend_timestamp_milliseconds": "",
                "status": "ok",
                "landmark_count": 33,
                "detail": "",
            }
            for fi, ts in sorted(_FRAME_TS.items())
        ]
        _write_csv(
            tmp_path / "pose_frames.csv",
            (
                "frame_index",
                "nominal_timestamp_seconds",
                "backend_timestamp_milliseconds",
                "status",
                "landmark_count",
                "detail",
            ),
            rows,
        )

    # gait_events.csv
    _write_csv(
        tmp_path / "gait_events.csv", GAIT_EVENT_FIELDS, [_event_row(e) for e in events]
    )

    # strides.csv
    _write_csv(
        tmp_path / "strides.csv", STRIDE_FIELDS, [_stride_row(s) for s in strides]
    )

    # gait_event_reviews.csv
    _write_csv(
        tmp_path / "gait_event_reviews.csv",
        (
            "event_id",
            "frame_index",
            "timestamp_seconds",
            "side",
            "detection_status",
            "review_status",
        ),
        event_reviews,
    )

    # strides_reviews.csv
    _write_csv(tmp_path / "strides_reviews.csv", STRIDE_FIELDS, stride_reviews)

    # gait_event_metadata.json
    config = GaitEventConfig()
    (tmp_path / "gait_event_metadata.json").write_text(
        json.dumps(
            {
                "config": {"detector": {"direction": config.direction}},
                "schema_version": 1,
                "algorithm_version": "step4-gait-event-detection-1",
            }
        )
        + "\n"
    )

    # preprocessing_metadata.json
    if include_preprocessing:
        pose_hash = hashlib.sha256(
            (tmp_path / "pose_frames.csv").read_bytes()
        ).hexdigest()
        meta: dict[str, Any] = {
            "inputs": {
                "pose_frames.csv": {
                    "path": str(tmp_path / "pose_frames.csv"),
                    "sha256": pose_hash,
                },
            },
        }
        if preprocessing_inputs:
            meta["inputs"].update(preprocessing_inputs)
        (tmp_path / "preprocessing_metadata.json").write_text(json.dumps(meta) + "\n")

    # assumption document
    (tmp_path / "assumptions.txt").write_text("test assumption document\n")
    return tmp_path


def _run_and_read(tmp_path: Path) -> ReviewResolutionArtifacts:
    result = resolve_gait_reviews(tmp_path, tmp_path / "assumptions.txt")
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Test: two-boundary end-to-end case with irregular timestamps
# ---------------------------------------------------------------------------


class TestTwoBoundaryEndToEnd:
    """Promoted replacement 82->86 and manual-only 138->140."""

    def test_full_resolution(self, tmp_path: Path) -> None:
        """Core scenario: left replacement + right correction + boundary checks."""
        # Left events: E0001(10), E0002(82), E0003(86 rejected), E0004(90)
        # Right events: E0005(30), E0006(138), E0007(160)
        events = [
            _make_event("E0001", 10, "left"),
            _make_event("E0002", 82, "left"),
            _make_event(
                "E0003",
                86,
                "left",
                status="rejected_candidate",
                included=False,
                quality="low",
            ),
            _make_event("E0004", 90, "left"),
            _make_event("E0005", 30, "right"),
            _make_event("E0006", 138, "right"),
            _make_event("E0007", 160, "right"),
        ]

        # Automatic strides (from construct_strides):
        # Left:  S0001(E0001->E0002), S0002(E0002->E0004)
        # Right: S0003(E0005->E0006), S0004(E0006->E0007)
        strides = [
            _make_stride("S0001", "left", "E0001", "E0002", 10, 82),
            _make_stride("S0002", "left", "E0002", "E0004", 82, 90),
            _make_stride("S0003", "right", "E0005", "E0006", 30, 138),
            _make_stride("S0004", "right", "E0006", "E0007", 138, 160),
        ]

        event_reviews = [
            {
                "event_id": "E0001",
                "frame_index": "10",
                "timestamp_seconds": str(_FRAME_TS[10]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0002",
                "frame_index": "82",
                "timestamp_seconds": str(_FRAME_TS[82]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0003",
                "frame_index": "86",
                "timestamp_seconds": str(_FRAME_TS[86]),
                "side": "left",
                "detection_status": "rejected_candidate",
                "review_status": "promote_to_candidate",
            },
            {
                "event_id": "E0004",
                "frame_index": "90",
                "timestamp_seconds": str(_FRAME_TS[90]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0005",
                "frame_index": "30",
                "timestamp_seconds": str(_FRAME_TS[30]),
                "side": "right",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0006",
                "frame_index": "138",
                "timestamp_seconds": str(_FRAME_TS[138]),
                "side": "right",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0007",
                "frame_index": "160",
                "timestamp_seconds": str(_FRAME_TS[160]),
                "side": "right",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
        ]

        # S0002 corrects start from 82 to 86 -> correction for E0002
        # S0004 corrects start from 138 to 140 -> correction for E0006
        # Stride reviews must have stale timestamps/durations matching automatic
        # strides, not values recomputed from corrected frames.
        auto_s0002_dur = _FRAME_TS[90] - _FRAME_TS[82]
        auto_s0004_dur = _FRAME_TS[160] - _FRAME_TS[138]
        stride_reviews = [
            _stride_row(
                _make_stride(
                    "S0001", "left", "E0001", "E0002", 10, 82, review_status="accept"
                )
            ),
            _stride_row(
                _make_stride(
                    "S0002",
                    "left",
                    "E0002",
                    "E0004",
                    86,
                    90,
                    review_status="correct",
                    start_timestamp_seconds=_FRAME_TS[82],
                    duration_seconds=auto_s0002_dur,
                )
            ),
            _stride_row(
                _make_stride(
                    "S0003", "right", "E0005", "E0006", 30, 138, review_status="accept"
                )
            ),
            _stride_row(
                _make_stride(
                    "S0004",
                    "right",
                    "E0006",
                    "E0007",
                    140,
                    160,
                    review_status="correct",
                    start_timestamp_seconds=_FRAME_TS[138],
                    duration_seconds=auto_s0004_dur,
                )
            ),
        ]

        _build_artifact_dir(
            tmp_path,
            events=events,
            strides=strides,
            event_reviews=event_reviews,
            stride_reviews=stride_reviews,
        )

        result = _run_and_read(tmp_path)
        assert result.reviewed_gait_events_path.is_file()
        assert result.reviewed_strides_path.is_file()
        assert result.review_resolution_metadata_path.is_file()

        rev_events = _read_csv(result.reviewed_gait_events_path)
        rev_strides = _read_csv(result.reviewed_strides_path)
        meta = _read_json(result.review_resolution_metadata_path)

        # --- Event assertions ---
        rev_by_auto = {r["automatic_event_id"]: r for r in rev_events}

        # E0002: replaced by E0003
        e2 = rev_by_auto["E0002"]
        assert e2["resolution_disposition"] == "replaced"
        assert e2["replaced_by_event_id"] == "E0003"
        assert e2["reviewed_accepted"] == "false"

        # E0003: promoted replacement
        e3 = rev_by_auto["E0003"]
        assert e3["resolution_disposition"] == "promoted_from_rejected_candidate"
        assert e3["reviewed_accepted"] == "true"
        assert e3["replaces_event_id"] == "E0002"
        assert e3["reviewed_quality"] == "review"
        assert e3["reviewed_frame_index"] == "86"
        assert float(e3["reviewed_timestamp_seconds"]) == pytest.approx(_FRAME_TS[86])
        # Provenance must identify source stride and boundary, not standalone
        assert e3["stride_review_provenance"] != "standalone_promotion"
        assert "S0002" in e3["stride_review_provenance"]
        assert "start" in e3["stride_review_provenance"]

        # E0006: corrected 138->140
        e6 = rev_by_auto["E0006"]
        assert e6["resolution_disposition"] == "corrected"
        assert e6["reviewed_frame_index"] == "140"
        assert float(e6["reviewed_timestamp_seconds"]) == pytest.approx(_FRAME_TS[140])
        assert e6["reviewed_quality"] == "review"
        assert e6["replaces_event_id"] == ""
        assert e6["replaced_by_event_id"] == ""
        # Finding 4: boundary direction in manual-only corrected provenance
        assert "start" in e6["stride_review_provenance"]
        assert "S0004" in e6["stride_review_provenance"]

        # E0001, E0004, E0005, E0007: unchanged
        for eid in ("E0001", "E0004", "E0005", "E0007"):
            assert rev_by_auto[eid]["resolution_disposition"] == "accepted_unchanged"
            assert rev_by_auto[eid]["reviewed_quality"] == "high"

        # --- Stride assertions ---
        strides_by_id = {s["stride_id"]: s for s in rev_strides}

        # S0001: E0001->E0003 (replacement changes end event ID)
        s1 = strides_by_id["S0001"]
        assert s1["start_event_id"] == "E0001"
        assert s1["end_event_id"] == "E0003"
        assert s1["start_frame"] == "10"
        assert s1["end_frame"] == "86"
        expected_dur_s1 = _FRAME_TS[86] - _FRAME_TS[10]
        assert float(s1["duration_seconds"]) == pytest.approx(expected_dur_s1)
        assert float(s1["duration_seconds"]) > 0

        # S0002: E0003->E0004 (replacement changes start event ID)
        s2 = strides_by_id["S0002"]
        assert s2["start_event_id"] == "E0003"
        assert s2["end_event_id"] == "E0004"
        assert s2["start_frame"] == "86"
        assert s2["end_frame"] == "90"
        expected_dur_s2 = _FRAME_TS[90] - _FRAME_TS[86]
        assert float(s2["duration_seconds"]) == pytest.approx(expected_dur_s2)

        # S0003: E0005->E0006 (corrected end frame)
        s3 = strides_by_id["S0003"]
        assert s3["start_event_id"] == "E0005"
        assert s3["end_event_id"] == "E0006"
        assert s3["start_frame"] == "30"
        assert s3["end_frame"] == "140"
        expected_dur_s3 = _FRAME_TS[140] - _FRAME_TS[30]
        assert float(s3["duration_seconds"]) == pytest.approx(expected_dur_s3)

        # S0004: E0006->E0007 (corrected start frame)
        s4 = strides_by_id["S0004"]
        assert s4["start_event_id"] == "E0006"
        assert s4["end_event_id"] == "E0007"
        assert s4["start_frame"] == "140"
        assert s4["end_frame"] == "160"
        expected_dur_s4 = _FRAME_TS[160] - _FRAME_TS[140]
        assert float(s4["duration_seconds"]) == pytest.approx(expected_dur_s4)

        # All durations positive
        for s in rev_strides:
            assert float(s["duration_seconds"]) > 0

        # Consecutive same-side shared boundaries
        left_strides = [s for s in rev_strides if s["side"] == "left"]
        right_strides = [s for s in rev_strides if s["side"] == "right"]
        assert left_strides[0]["end_event_id"] == left_strides[1]["start_event_id"]
        assert left_strides[0]["end_frame"] == left_strides[1]["start_frame"]
        assert right_strides[0]["end_event_id"] == right_strides[1]["start_event_id"]
        assert right_strides[0]["end_frame"] == right_strides[1]["start_frame"]

        # --- Metadata assertions ---
        assert meta["schema_version"] == REVIEW_RESOLUTION_SCHEMA_VERSION
        assert meta["algorithm_version"] == REVIEW_RESOLUTION_ALGORITHM_VERSION
        assert meta["counts"]["automatic_events"] == 7
        assert meta["counts"]["reviewed_accepted"] == 6
        assert meta["counts"]["accepted_unchanged"] == 4
        assert meta["counts"]["corrected"] == 1
        assert meta["counts"]["promoted"] == 1
        assert meta["counts"]["replaced"] == 1
        assert meta["counts"]["reviewed_strides"] == 4
        assert meta["counts"]["rejected"] == 1

        # Boundary changes: reviewed_event_id + reviewed_timestamp
        bc_by_auto = {b["automatic_event_id"]: b for b in meta["boundary_changes"]}
        # E0002 correction: replaced, reviewed_event_id=E0003
        bc_e2 = bc_by_auto["E0002"]
        assert bc_e2["reviewed_event_id"] == "E0003"
        assert bc_e2["reviewed_frame"] == 86
        assert bc_e2["reviewed_timestamp"] == pytest.approx(_FRAME_TS[86])
        assert bc_e2["promotion_provenance"] == "promoted_candidate:E0003"
        # E0006 correction: manual-only, reviewed_event_id=E0006
        bc_e6 = bc_by_auto["E0006"]
        assert bc_e6["reviewed_event_id"] == "E0006"
        assert bc_e6["reviewed_frame"] == 140
        assert bc_e6["reviewed_timestamp"] == pytest.approx(_FRAME_TS[140])
        assert bc_e6["promotion_provenance"] is None

        # Checks performed (deduplicated sorted)
        assert isinstance(meta["checks_performed"], list)
        assert len(meta["checks_performed"]) == len(set(meta["checks_performed"]))
        assert meta["checks_performed"] == sorted(meta["checks_performed"])
        assert "endpoint_event_id_side_frame" in meta["checks_performed"]
        assert "endpoint_timestamps" in meta["checks_performed"]
        assert "duration_positive" in meta["checks_performed"]
        assert "accepted_endpoints" in meta["checks_performed"]
        assert "correction_propagated" in meta["checks_performed"]
        assert "consecutive_shared_boundary" in meta["checks_performed"]

        # Timestamp source
        assert "timestamp_source" in meta
        assert meta["timestamp_source"]["sha256"]

        # Assumption document: hashed provenance only
        assumption_rec = meta["inputs"]["assumptions.txt"]
        assert "sha256" in assumption_rec
        assert "not machine-evaluated" in assumption_rec["note"]

        # Scientific unresolved
        assert isinstance(meta["scientific_unresolved"], list)
        assert len(meta["scientific_unresolved"]) > 0
        assert any("manually interpreted" in s for s in meta["scientific_unresolved"])

        # Source Step4 linkage
        s4 = meta["source_step4"]
        assert "gait_event_config" in s4
        assert "schema_version" in s4
        assert "algorithm_version" in s4
        assert s4["schema_version"] == 1
        assert s4["algorithm_version"] == "step4-gait-event-detection-1"

        # Limitations
        assert any("COM" in lim or "step5" in lim for lim in meta["limitations"])


# ---------------------------------------------------------------------------
# Test: retained/baseline rejected candidates excluded
# ---------------------------------------------------------------------------


class TestRejectedExclusion:
    def test_retained_rejection_excluded(self, tmp_path: Path) -> None:
        events = [
            _make_event("E0001", 10, "left"),
            _make_event(
                "E0002",
                30,
                "left",
                status="rejected_candidate",
                included=False,
                quality="low",
            ),
            _make_event("E0003", 50, "left"),
        ]
        strides = [_make_stride("S0001", "left", "E0001", "E0003", 10, 50)]
        event_reviews = [
            {
                "event_id": "E0001",
                "frame_index": "10",
                "timestamp_seconds": str(_FRAME_TS[10]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0002",
                "frame_index": "30",
                "timestamp_seconds": str(_FRAME_TS[30]),
                "side": "left",
                "detection_status": "rejected_candidate",
                "review_status": "retain_rejection",
            },
            {
                "event_id": "E0003",
                "frame_index": "50",
                "timestamp_seconds": str(_FRAME_TS[50]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
        ]
        stride_reviews = [
            _stride_row(
                _make_stride(
                    "S0001", "left", "E0001", "E0003", 10, 50, review_status="accept"
                )
            )
        ]
        _build_artifact_dir(
            tmp_path,
            events=events,
            strides=strides,
            event_reviews=event_reviews,
            stride_reviews=stride_reviews,
        )
        result = _run_and_read(tmp_path)
        rev_events = _read_csv(result.reviewed_gait_events_path)
        rev_by_auto = {r["automatic_event_id"]: r for r in rev_events}

        # E0002 remains rejected
        assert rev_by_auto["E0002"]["resolution_disposition"] == "rejected"
        assert rev_by_auto["E0002"]["reviewed_accepted"] == "false"
        # E0001, E0003 unchanged
        assert rev_by_auto["E0001"]["resolution_disposition"] == "accepted_unchanged"
        assert rev_by_auto["E0003"]["resolution_disposition"] == "accepted_unchanged"


# ---------------------------------------------------------------------------
# Test: all 8 input hashes unchanged
# ---------------------------------------------------------------------------


class TestInputHashUnchanged:
    def test_hashes_unchanged(self, tmp_path: Path) -> None:
        events = [
            _make_event("E0001", 10, "left"),
            _make_event("E0002", 50, "left"),
        ]
        strides = [_make_stride("S0001", "left", "E0001", "E0002", 10, 50)]
        event_reviews = [
            {
                "event_id": "E0001",
                "frame_index": "10",
                "timestamp_seconds": str(_FRAME_TS[10]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0002",
                "frame_index": "50",
                "timestamp_seconds": str(_FRAME_TS[50]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
        ]
        stride_reviews = [
            _stride_row(
                _make_stride(
                    "S0001", "left", "E0001", "E0002", 10, 50, review_status="accept"
                )
            )
        ]
        _build_artifact_dir(
            tmp_path,
            events=events,
            strides=strides,
            event_reviews=event_reviews,
            stride_reviews=stride_reviews,
        )

        # Snapshot input hashes before
        pre_hashes = {}
        for name in INPUT_ARTIFACT_NAMES:
            p = tmp_path / name
            if p.is_file():
                pre_hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        pre_hashes["assumptions.txt"] = hashlib.sha256(
            (tmp_path / "assumptions.txt").read_bytes()
        ).hexdigest()

        _run_and_read(tmp_path)

        # Verify all input hashes unchanged
        for name in INPUT_ARTIFACT_NAMES:
            p = tmp_path / name
            if p.is_file():
                assert hashlib.sha256(p.read_bytes()).hexdigest() == pre_hashes[name]
        assert (
            hashlib.sha256((tmp_path / "assumptions.txt").read_bytes()).hexdigest()
            == pre_hashes["assumptions.txt"]
        )


# ---------------------------------------------------------------------------
# Test: rerun replaces only Step4b outputs
# ---------------------------------------------------------------------------


class TestRerunReplacement:
    def test_rerun_overwrites_outputs(self, tmp_path: Path) -> None:
        events = [
            _make_event("E0001", 10, "left"),
            _make_event("E0002", 50, "left"),
        ]
        strides = [_make_stride("S0001", "left", "E0001", "E0002", 10, 50)]
        event_reviews = [
            {
                "event_id": "E0001",
                "frame_index": "10",
                "timestamp_seconds": str(_FRAME_TS[10]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0002",
                "frame_index": "50",
                "timestamp_seconds": str(_FRAME_TS[50]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
        ]
        stride_reviews = [
            _stride_row(
                _make_stride(
                    "S0001", "left", "E0001", "E0002", 10, 50, review_status="accept"
                )
            )
        ]
        _build_artifact_dir(
            tmp_path,
            events=events,
            strides=strides,
            event_reviews=event_reviews,
            stride_reviews=stride_reviews,
        )

        # First run
        r1 = _run_and_read(tmp_path)
        h1_events = hashlib.sha256(
            r1.reviewed_gait_events_path.read_bytes()
        ).hexdigest()
        h1_strides = hashlib.sha256(r1.reviewed_strides_path.read_bytes()).hexdigest()

        # Create a dummy file that should NOT be touched
        dummy = tmp_path / "reviewed_extra.txt"
        dummy.write_text("should survive")

        # Second run
        r2 = _run_and_read(tmp_path)
        # Same content (deterministic)
        assert (
            hashlib.sha256(r2.reviewed_gait_events_path.read_bytes()).hexdigest()
            == h1_events
        )
        assert (
            hashlib.sha256(r2.reviewed_strides_path.read_bytes()).hexdigest()
            == h1_strides
        )
        # Dummy file survived
        assert dummy.read_text() == "should survive"


# ---------------------------------------------------------------------------
# Test: failures with no partial outputs
# ---------------------------------------------------------------------------


class TestFailureCases:
    def _base_events(self) -> tuple[list[GaitEvent], list[Stride]]:
        events = [
            _make_event("E0001", 10, "left"),
            _make_event("E0002", 50, "left"),
            _make_event(
                "E0003",
                50,
                "right",
                status="rejected_candidate",
                included=False,
                quality="low",
            ),
            _make_event("E0004", 80, "right"),
            _make_event("E0005", 120, "right"),
        ]
        strides = [
            _make_stride("S0001", "left", "E0001", "E0002", 10, 50),
            _make_stride("S0002", "right", "E0004", "E0005", 80, 120),
        ]
        return events, strides

    def _base_reviews(
        self,
        *,
        e3_review: str = "unreviewed",
        s2_start: int = 80,
        s2_review: str = "accept",
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        event_reviews = [
            {
                "event_id": "E0001",
                "frame_index": "10",
                "timestamp_seconds": str(_FRAME_TS[10]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0002",
                "frame_index": "50",
                "timestamp_seconds": str(_FRAME_TS[50]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0003",
                "frame_index": "50",
                "timestamp_seconds": str(_FRAME_TS[50]),
                "side": "right",
                "detection_status": "rejected_candidate",
                "review_status": e3_review,
            },
            {
                "event_id": "E0004",
                "frame_index": "80",
                "timestamp_seconds": str(_FRAME_TS[80]),
                "side": "right",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0005",
                "frame_index": "120",
                "timestamp_seconds": str(_FRAME_TS[120]),
                "side": "right",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
        ]
        stride_reviews = [
            _stride_row(
                _make_stride(
                    "S0001", "left", "E0001", "E0002", 10, 50, review_status="accept"
                )
            ),
            _stride_row(
                _make_stride(
                    "S0002",
                    "right",
                    "E0004",
                    "E0005",
                    s2_start,
                    120,
                    review_status=s2_review,
                )
            ),
        ]
        return event_reviews, stride_reviews

    def test_candidate_at_corrected_frame_not_promoted(self, tmp_path: Path) -> None:
        """Corrected frame has a same-side rejected candidate that is NOT promoted."""
        # E0001: left, 10 (accepted)
        # E0002: left, 50 (accepted) — correction target: end corrected 50 -> 80
        # E0005: left, 80 (rejected_candidate, NOT promoted) — same side as E0002
        # Automatic strides only use accepted/included endpoints.
        events = [
            _make_event("E0001", 10, "left"),
            _make_event("E0002", 50, "left"),
            _make_event(
                "E0005",
                80,
                "left",
                status="rejected_candidate",
                included=False,
                quality="low",
            ),
        ]
        strides = [
            _make_stride("S0001", "left", "E0001", "E0002", 10, 50),
        ]

        event_reviews = [
            {
                "event_id": "E0001",
                "frame_index": "10",
                "timestamp_seconds": str(_FRAME_TS[10]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0002",
                "frame_index": "50",
                "timestamp_seconds": str(_FRAME_TS[50]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0005",
                "frame_index": "80",
                "timestamp_seconds": str(_FRAME_TS[80]),
                "side": "left",
                "detection_status": "rejected_candidate",
                "review_status": "unreviewed",
            },
        ]
        # S0001 corrects end from 50 to 80 — correction for E0002.
        # E0005 at frame 80 (same side, rejected, NOT promoted) should
        # cause failure: "not explicitly promoted".
        auto_s0001_dur = _FRAME_TS[50] - _FRAME_TS[10]
        stride_reviews = [
            _stride_row(
                _make_stride(
                    "S0001",
                    "left",
                    "E0001",
                    "E0002",
                    10,
                    80,
                    review_status="correct",
                    end_timestamp_seconds=_FRAME_TS[50],
                    duration_seconds=auto_s0001_dur,
                )
            ),
        ]
        _build_artifact_dir(
            tmp_path,
            events=events,
            strides=strides,
            event_reviews=event_reviews,
            stride_reviews=stride_reviews,
        )
        with pytest.raises(
            ReviewResolutionArtifactValidationError, match="not.*explicitly promoted"
        ):
            resolve_gait_reviews(tmp_path, tmp_path / "assumptions.txt")

    def test_wrong_side_candidate_at_corrected_frame(self, tmp_path: Path) -> None:
        """Corrected frame has a wrong-side accepted candidate."""
        events, strides = self._base_events()
        event_reviews, stride_reviews = self._base_reviews()

        # Correction for E0002 to frame 80 where E0004 (right, accepted) exists
        # Stride review must have stale timestamps/durations matching automatic.
        auto_s0001_dur = _FRAME_TS[50] - _FRAME_TS[10]
        stride_reviews[0] = _stride_row(
            _make_stride(
                "S0001",
                "left",
                "E0001",
                "E0002",
                80,
                50,
                review_status="correct",
                start_timestamp_seconds=_FRAME_TS[10],
                duration_seconds=auto_s0001_dur,
            )
        )
        _build_artifact_dir(
            tmp_path,
            events=events,
            strides=strides,
            event_reviews=event_reviews,
            stride_reviews=stride_reviews,
        )
        with pytest.raises(
            ReviewResolutionArtifactValidationError, match="wrong-side candidates"
        ):
            resolve_gait_reviews(tmp_path, tmp_path / "assumptions.txt")

    def test_accept_with_frame_edit_rejected(self, tmp_path: Path) -> None:
        """Review status 'accept' with frame edits should fail."""
        events = [
            _make_event("E0001", 10, "left"),
            _make_event("E0002", 50, "left"),
        ]
        strides = [_make_stride("S0001", "left", "E0001", "E0002", 10, 50)]
        event_reviews = [
            {
                "event_id": "E0001",
                "frame_index": "10",
                "timestamp_seconds": str(_FRAME_TS[10]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0002",
                "frame_index": "50",
                "timestamp_seconds": str(_FRAME_TS[50]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
        ]
        # accept but with frame edit
        stride_reviews = [
            _stride_row(
                _make_stride(
                    "S0001", "left", "E0001", "E0002", 60, 50, review_status="accept"
                )
            )
        ]
        _build_artifact_dir(
            tmp_path,
            events=events,
            strides=strides,
            event_reviews=event_reviews,
            stride_reviews=stride_reviews,
        )
        with pytest.raises(
            ReviewResolutionArtifactValidationError,
            match="frame edits require review_status correct",
        ):
            resolve_gait_reviews(tmp_path, tmp_path / "assumptions.txt")

    def test_correct_with_no_frame_edit_rejected(self, tmp_path: Path) -> None:
        """Review status 'correct' with no frame edits should fail."""
        events = [
            _make_event("E0001", 10, "left"),
            _make_event("E0002", 50, "left"),
        ]
        strides = [_make_stride("S0001", "left", "E0001", "E0002", 10, 50)]
        event_reviews = [
            {
                "event_id": "E0001",
                "frame_index": "10",
                "timestamp_seconds": str(_FRAME_TS[10]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0002",
                "frame_index": "50",
                "timestamp_seconds": str(_FRAME_TS[50]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
        ]
        # correct but same frames
        stride_reviews = [
            _stride_row(
                _make_stride(
                    "S0001", "left", "E0001", "E0002", 10, 50, review_status="correct"
                )
            )
        ]
        _build_artifact_dir(
            tmp_path,
            events=events,
            strides=strides,
            event_reviews=event_reviews,
            stride_reviews=stride_reviews,
        )
        with pytest.raises(
            ReviewResolutionArtifactValidationError,
            match="no frame edits require review_status accept",
        ):
            resolve_gait_reviews(tmp_path, tmp_path / "assumptions.txt")

    def test_duplicate_review_row_rejected(self, tmp_path: Path) -> None:
        events = [
            _make_event("E0001", 10, "left"),
            _make_event("E0002", 50, "left"),
        ]
        strides = [_make_stride("S0001", "left", "E0001", "E0002", 10, 50)]
        base_event_review = {
            "event_id": "E0001",
            "frame_index": "10",
            "timestamp_seconds": str(_FRAME_TS[10]),
            "side": "left",
            "detection_status": "accepted",
            "review_status": "unreviewed",
        }
        event_reviews = [
            base_event_review,
            base_event_review,  # duplicate
            {
                "event_id": "E0002",
                "frame_index": "50",
                "timestamp_seconds": str(_FRAME_TS[50]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
        ]
        stride_reviews = [
            _stride_row(
                _make_stride(
                    "S0001", "left", "E0001", "E0002", 10, 50, review_status="accept"
                )
            )
        ]
        _build_artifact_dir(
            tmp_path,
            events=events,
            strides=strides,
            event_reviews=event_reviews,
            stride_reviews=stride_reviews,
        )
        with pytest.raises(
            ReviewResolutionArtifactValidationError, match="duplicate event_id"
        ):
            resolve_gait_reviews(tmp_path, tmp_path / "assumptions.txt")

    def test_missing_review_row_rejected(self, tmp_path: Path) -> None:
        events = [
            _make_event("E0001", 10, "left"),
            _make_event("E0002", 50, "left"),
        ]
        strides = [_make_stride("S0001", "left", "E0001", "E0002", 10, 50)]
        # Missing E0002
        event_reviews = [
            {
                "event_id": "E0001",
                "frame_index": "10",
                "timestamp_seconds": str(_FRAME_TS[10]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
        ]
        stride_reviews = [
            _stride_row(
                _make_stride(
                    "S0001", "left", "E0001", "E0002", 10, 50, review_status="accept"
                )
            )
        ]
        _build_artifact_dir(
            tmp_path,
            events=events,
            strides=strides,
            event_reviews=event_reviews,
            stride_reviews=stride_reviews,
        )
        with pytest.raises(
            ReviewResolutionArtifactValidationError, match="missing rows"
        ):
            resolve_gait_reviews(tmp_path, tmp_path / "assumptions.txt")

    def test_auto_timestamp_mismatch_rejected(self, tmp_path: Path) -> None:
        events = [
            _make_event("E0001", 10, "left"),
            _make_event("E0002", 50, "left"),
        ]
        strides = [_make_stride("S0001", "left", "E0001", "E0002", 10, 50)]
        event_reviews = [
            # Wrong timestamp for E0001
            {
                "event_id": "E0001",
                "frame_index": "10",
                "timestamp_seconds": "999.0",
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0002",
                "frame_index": "50",
                "timestamp_seconds": str(_FRAME_TS[50]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
        ]
        stride_reviews = [
            _stride_row(
                _make_stride(
                    "S0001", "left", "E0001", "E0002", 10, 50, review_status="accept"
                )
            )
        ]
        _build_artifact_dir(
            tmp_path,
            events=events,
            strides=strides,
            event_reviews=event_reviews,
            stride_reviews=stride_reviews,
        )
        with pytest.raises(
            ReviewResolutionArtifactValidationError, match="timestamp.*does not match"
        ):
            resolve_gait_reviews(tmp_path, tmp_path / "assumptions.txt")


# ---------------------------------------------------------------------------
# Test: preprocessing metadata must contain pose_frames hash
# ---------------------------------------------------------------------------


class TestPreprocessingHash:
    def test_missing_hash_rejected(self, tmp_path: Path) -> None:
        events = [
            _make_event("E0001", 10, "left"),
            _make_event("E0002", 50, "left"),
        ]
        strides = [_make_stride("S0001", "left", "E0001", "E0002", 10, 50)]
        event_reviews = [
            {
                "event_id": "E0001",
                "frame_index": "10",
                "timestamp_seconds": str(_FRAME_TS[10]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0002",
                "frame_index": "50",
                "timestamp_seconds": str(_FRAME_TS[50]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
        ]
        stride_reviews = [
            _stride_row(
                _make_stride(
                    "S0001", "left", "E0001", "E0002", 10, 50, review_status="accept"
                )
            )
        ]
        _build_artifact_dir(
            tmp_path,
            events=events,
            strides=strides,
            event_reviews=event_reviews,
            stride_reviews=stride_reviews,
        )
        # Overwrite preprocessing metadata without pose_frames hash
        (tmp_path / "preprocessing_metadata.json").write_text(
            json.dumps({"inputs": {}}) + "\n"
        )
        with pytest.raises(
            ReviewResolutionArtifactValidationError,
            match="must contain inputs.pose_frames.csv.sha256",
        ):
            resolve_gait_reviews(tmp_path, tmp_path / "assumptions.txt")

    def test_null_hash_rejected(self, tmp_path: Path) -> None:
        events = [
            _make_event("E0001", 10, "left"),
            _make_event("E0002", 50, "left"),
        ]
        strides = [_make_stride("S0001", "left", "E0001", "E0002", 10, 50)]
        event_reviews = [
            {
                "event_id": "E0001",
                "frame_index": "10",
                "timestamp_seconds": str(_FRAME_TS[10]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0002",
                "frame_index": "50",
                "timestamp_seconds": str(_FRAME_TS[50]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
        ]
        stride_reviews = [
            _stride_row(
                _make_stride(
                    "S0001", "left", "E0001", "E0002", 10, 50, review_status="accept"
                )
            )
        ]
        _build_artifact_dir(
            tmp_path,
            events=events,
            strides=strides,
            event_reviews=event_reviews,
            stride_reviews=stride_reviews,
        )
        (tmp_path / "preprocessing_metadata.json").write_text(
            json.dumps({"inputs": {"pose_frames.csv": {"sha256": None}}}) + "\n"
        )
        with pytest.raises(
            ReviewResolutionArtifactValidationError,
            match="must contain inputs.pose_frames.csv.sha256",
        ):
            resolve_gait_reviews(tmp_path, tmp_path / "assumptions.txt")


# ---------------------------------------------------------------------------
# Test: cross-side target frame regression (Finding 1)
# ---------------------------------------------------------------------------


class TestCrossSideTargetFrame:
    """Two corrections target the same frame on different sides; must not cross-link."""

    def test_no_wrong_side_cross_link(self, tmp_path: Path) -> None:
        """Left correction to frame 80 and right correction to frame 80
        must not allow a promoted left candidate to be treated as a
        replacement for the right correction or vice versa."""
        # Left events: E0001(10), E0002(50), E0003(80 rejected), E0004(90)
        # Right events: E0005(10), E0006(50), E0007(80 rejected), E0008(90)
        events = [
            _make_event("E0001", 10, "left"),
            _make_event("E0002", 50, "left"),
            _make_event(
                "E0003",
                80,
                "left",
                status="rejected_candidate",
                included=False,
                quality="low",
            ),
            _make_event("E0004", 90, "left"),
            _make_event("E0005", 10, "right"),
            _make_event("E0006", 50, "right"),
            _make_event(
                "E0007",
                80,
                "right",
                status="rejected_candidate",
                included=False,
                quality="low",
            ),
            _make_event("E0008", 90, "right"),
        ]
        strides = [
            _make_stride("SL1", "left", "E0001", "E0002", 10, 50),
            _make_stride("SL2", "left", "E0002", "E0004", 50, 90),
            _make_stride("SR1", "right", "E0005", "E0006", 10, 50),
            _make_stride("SR2", "right", "E0006", "E0008", 50, 90),
        ]
        event_reviews = [
            {
                "event_id": "E0001",
                "frame_index": "10",
                "timestamp_seconds": str(_FRAME_TS[10]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0002",
                "frame_index": "50",
                "timestamp_seconds": str(_FRAME_TS[50]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0003",
                "frame_index": "80",
                "timestamp_seconds": str(_FRAME_TS[80]),
                "side": "left",
                "detection_status": "rejected_candidate",
                "review_status": "promote_to_candidate",
            },
            {
                "event_id": "E0004",
                "frame_index": "90",
                "timestamp_seconds": str(_FRAME_TS[90]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0005",
                "frame_index": "10",
                "timestamp_seconds": str(_FRAME_TS[10]),
                "side": "right",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0006",
                "frame_index": "50",
                "timestamp_seconds": str(_FRAME_TS[50]),
                "side": "right",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0007",
                "frame_index": "80",
                "timestamp_seconds": str(_FRAME_TS[80]),
                "side": "right",
                "detection_status": "rejected_candidate",
                "review_status": "promote_to_candidate",
            },
            {
                "event_id": "E0008",
                "frame_index": "90",
                "timestamp_seconds": str(_FRAME_TS[90]),
                "side": "right",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
        ]
        # Both strides corrected to frame 80 (same frame, different sides)
        # Stride reviews must have stale timestamps/durations matching automatic
        # strides, not values recomputed from corrected frames.
        auto_sl2_dur = _FRAME_TS[90] - _FRAME_TS[50]
        auto_sr2_dur = _FRAME_TS[90] - _FRAME_TS[50]
        stride_reviews = [
            _stride_row(
                _make_stride(
                    "SL1", "left", "E0001", "E0002", 10, 50, review_status="accept"
                )
            ),
            _stride_row(
                _make_stride(
                    "SL2",
                    "left",
                    "E0002",
                    "E0004",
                    80,
                    90,
                    review_status="correct",
                    start_timestamp_seconds=_FRAME_TS[50],
                    duration_seconds=auto_sl2_dur,
                )
            ),
            _stride_row(
                _make_stride(
                    "SR1", "right", "E0005", "E0006", 10, 50, review_status="accept"
                )
            ),
            _stride_row(
                _make_stride(
                    "SR2",
                    "right",
                    "E0006",
                    "E0008",
                    80,
                    90,
                    review_status="correct",
                    start_timestamp_seconds=_FRAME_TS[50],
                    duration_seconds=auto_sr2_dur,
                )
            ),
        ]
        _build_artifact_dir(
            tmp_path,
            events=events,
            strides=strides,
            event_reviews=event_reviews,
            stride_reviews=stride_reviews,
        )
        result = _run_and_read(tmp_path)
        rev_events = _read_csv(result.reviewed_gait_events_path)
        rev_by_auto = {r["automatic_event_id"]: r for r in rev_events}

        # Left promoted candidate E0003 replaces left E0002 only
        assert rev_by_auto["E0003"]["replaces_event_id"] == "E0002"
        assert rev_by_auto["E0003"]["side"] == "left"
        # Right promoted candidate E0007 replaces right E0006 only
        assert rev_by_auto["E0007"]["replaces_event_id"] == "E0006"
        assert rev_by_auto["E0007"]["side"] == "right"
        # Neither cross-links
        assert rev_by_auto["E0003"]["replaces_event_id"] != "E0006"
        assert rev_by_auto["E0007"]["replaces_event_id"] != "E0002"


# ---------------------------------------------------------------------------
# Test: strides_reviews.csv field corruption rejection (Finding 5)
# ---------------------------------------------------------------------------


class TestStrideReviewFieldCorruption:
    def test_corrupted_quality_rejected(self, tmp_path: Path) -> None:
        """Quality field in strides_reviews.csv must match automatic stride."""
        events = [
            _make_event("E0001", 10, "left"),
            _make_event("E0002", 50, "left"),
        ]
        strides = [_make_stride("S0001", "left", "E0001", "E0002", 10, 50)]
        event_reviews = [
            {
                "event_id": "E0001",
                "frame_index": "10",
                "timestamp_seconds": str(_FRAME_TS[10]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0002",
                "frame_index": "50",
                "timestamp_seconds": str(_FRAME_TS[50]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
        ]
        # quality is "low" but automatic stride has "high"
        stride_reviews = [
            _stride_row(
                _make_stride(
                    "S0001",
                    "left",
                    "E0001",
                    "E0002",
                    10,
                    50,
                    review_status="accept",
                    quality="low",
                )
            )
        ]
        _build_artifact_dir(
            tmp_path,
            events=events,
            strides=strides,
            event_reviews=event_reviews,
            stride_reviews=stride_reviews,
        )
        with pytest.raises(
            ReviewResolutionArtifactValidationError, match="quality.*does not match"
        ):
            resolve_gait_reviews(tmp_path, tmp_path / "assumptions.txt")

    def test_corrupted_source_rejected(self, tmp_path: Path) -> None:
        """Source field in strides_reviews.csv must match automatic stride."""
        events = [
            _make_event("E0001", 10, "left"),
            _make_event("E0002", 50, "left"),
        ]
        strides = [_make_stride("S0001", "left", "E0001", "E0002", 10, 50)]
        event_reviews = [
            {
                "event_id": "E0001",
                "frame_index": "10",
                "timestamp_seconds": str(_FRAME_TS[10]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0002",
                "frame_index": "50",
                "timestamp_seconds": str(_FRAME_TS[50]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
        ]
        stride_reviews = [
            _stride_row(
                _make_stride(
                    "S0001",
                    "left",
                    "E0001",
                    "E0002",
                    10,
                    50,
                    review_status="accept",
                    source="manual_override",
                )
            )
        ]
        _build_artifact_dir(
            tmp_path,
            events=events,
            strides=strides,
            event_reviews=event_reviews,
            stride_reviews=stride_reviews,
        )
        with pytest.raises(
            ReviewResolutionArtifactValidationError, match="source.*does not match"
        ):
            resolve_gait_reviews(tmp_path, tmp_path / "assumptions.txt")

    def test_corrupted_start_timestamp_rejected(self, tmp_path: Path) -> None:
        """Stale start_timestamp_seconds in strides_reviews.csv must match
        automatic stride to prevent silent data corruption."""
        events = [
            _make_event("E0001", 10, "left"),
            _make_event("E0002", 50, "left"),
        ]
        strides = [_make_stride("S0001", "left", "E0001", "E0002", 10, 50)]
        event_reviews = [
            {
                "event_id": "E0001",
                "frame_index": "10",
                "timestamp_seconds": str(_FRAME_TS[10]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0002",
                "frame_index": "50",
                "timestamp_seconds": str(_FRAME_TS[50]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
        ]
        # start_timestamp_seconds is wrong (999.0 vs expected _FRAME_TS[10])
        stride_reviews = [
            _stride_row(
                _make_stride(
                    "S0001",
                    "left",
                    "E0001",
                    "E0002",
                    10,
                    50,
                    review_status="accept",
                    start_timestamp_seconds=999.0,
                )
            )
        ]
        _build_artifact_dir(
            tmp_path,
            events=events,
            strides=strides,
            event_reviews=event_reviews,
            stride_reviews=stride_reviews,
        )
        with pytest.raises(
            ReviewResolutionArtifactValidationError,
            match="start_timestamp_seconds.*does not match",
        ):
            resolve_gait_reviews(tmp_path, tmp_path / "assumptions.txt")

    def test_corrupted_duration_rejected(self, tmp_path: Path) -> None:
        """Stale duration_seconds in strides_reviews.csv must match
        automatic stride to prevent silent data corruption."""
        events = [
            _make_event("E0001", 10, "left"),
            _make_event("E0002", 50, "left"),
        ]
        strides = [_make_stride("S0001", "left", "E0001", "E0002", 10, 50)]
        event_reviews = [
            {
                "event_id": "E0001",
                "frame_index": "10",
                "timestamp_seconds": str(_FRAME_TS[10]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0002",
                "frame_index": "50",
                "timestamp_seconds": str(_FRAME_TS[50]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
        ]
        # duration_seconds is wrong (999.0 vs expected ~1.634)
        stride_reviews = [
            _stride_row(
                _make_stride(
                    "S0001",
                    "left",
                    "E0001",
                    "E0002",
                    10,
                    50,
                    review_status="accept",
                    duration_seconds=999.0,
                )
            )
        ]
        _build_artifact_dir(
            tmp_path,
            events=events,
            strides=strides,
            event_reviews=event_reviews,
            stride_reviews=stride_reviews,
        )
        with pytest.raises(
            ReviewResolutionArtifactValidationError,
            match="duration_seconds.*does not match",
        ):
            resolve_gait_reviews(tmp_path, tmp_path / "assumptions.txt")


# ---------------------------------------------------------------------------
# Test: standalone promotion rejection (Finding 6)
# ---------------------------------------------------------------------------


class TestStandalonePromotionRejection:
    def test_unlinked_promotion_rejected(self, tmp_path: Path) -> None:
        """A promote_to_candidate not linked as the target replacement of any
        event-level stride boundary correction must fail.

        Input contract is valid: automatic strides contain only A->C with
        accepted/included endpoints.  Candidate B (rejected, excluded) sits
        temporally between A and C but is not used by any automatic stride.
        Event review promotes B standalone; stride review accepts A->C
        unchanged.  Must fail as unsupported standalone promotion.
        """
        # A(accepted, left, 10) -> C(accepted, left, 80) is the only stride.
        # B(rejected, left, 50) is between A and C but excluded from strides.
        events = [
            _make_event("E0001", 10, "left"),
            _make_event(
                "E0002",
                50,
                "left",
                status="rejected_candidate",
                included=False,
                quality="low",
            ),
            _make_event("E0003", 80, "left"),
        ]
        # Only one automatic stride: A->C (no stride uses B).
        strides = [_make_stride("S0001", "left", "E0001", "E0003", 10, 80)]
        event_reviews = [
            {
                "event_id": "E0001",
                "frame_index": "10",
                "timestamp_seconds": str(_FRAME_TS[10]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0002",
                "frame_index": "50",
                "timestamp_seconds": str(_FRAME_TS[50]),
                "side": "left",
                "detection_status": "rejected_candidate",
                "review_status": "promote_to_candidate",
            },
            {
                "event_id": "E0003",
                "frame_index": "80",
                "timestamp_seconds": str(_FRAME_TS[80]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
        ]
        # Stride review accepts A->C unchanged; no boundary correction links
        # B as a replacement target.
        stride_reviews = [
            _stride_row(
                _make_stride(
                    "S0001", "left", "E0001", "E0003", 10, 80, review_status="accept"
                )
            ),
        ]
        _build_artifact_dir(
            tmp_path,
            events=events,
            strides=strides,
            event_reviews=event_reviews,
            stride_reviews=stride_reviews,
        )
        with pytest.raises(
            ReviewResolutionArtifactValidationError,
            match="standalone promotions are not supported",
        ):
            resolve_gait_reviews(tmp_path, tmp_path / "assumptions.txt")

    def test_automatic_stride_with_rejected_endpoint_rejected(
        self, tmp_path: Path
    ) -> None:
        """An automatic stride whose endpoint is a rejected_candidate must be
        rejected by input validation."""
        # E0001 accepted, E0002 rejected, E0003 accepted.
        # Stride S0001 erroneously uses rejected E0002 as end_event.
        events = [
            _make_event("E0001", 10, "left"),
            _make_event(
                "E0002",
                50,
                "left",
                status="rejected_candidate",
                included=False,
                quality="low",
            ),
            _make_event("E0003", 80, "left"),
        ]
        strides = [_make_stride("S0001", "left", "E0001", "E0002", 10, 50)]
        event_reviews = [
            {
                "event_id": "E0001",
                "frame_index": "10",
                "timestamp_seconds": str(_FRAME_TS[10]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0002",
                "frame_index": "50",
                "timestamp_seconds": str(_FRAME_TS[50]),
                "side": "left",
                "detection_status": "rejected_candidate",
                "review_status": "unreviewed",
            },
            {
                "event_id": "E0003",
                "frame_index": "80",
                "timestamp_seconds": str(_FRAME_TS[80]),
                "side": "left",
                "detection_status": "accepted",
                "review_status": "unreviewed",
            },
        ]
        stride_reviews = [
            _stride_row(
                _make_stride(
                    "S0001", "left", "E0001", "E0002", 10, 50, review_status="accept"
                )
            ),
        ]
        _build_artifact_dir(
            tmp_path,
            events=events,
            strides=strides,
            event_reviews=event_reviews,
            stride_reviews=stride_reviews,
        )
        with pytest.raises(
            ReviewResolutionArtifactValidationError,
            match="detection_status.*is not accepted",
        ):
            resolve_gait_reviews(tmp_path, tmp_path / "assumptions.txt")


# ---------------------------------------------------------------------------
# Test: public exports
# ---------------------------------------------------------------------------


class TestPublicExports:
    def test_error_hierarchy(self) -> None:
        assert issubclass(
            ReviewResolutionArtifactValidationError, ReviewResolutionError
        )
        assert issubclass(ReviewResolutionError, Exception)

    def test_constants(self) -> None:
        assert REVIEW_RESOLUTION_SCHEMA_VERSION == 1
        assert isinstance(REVIEW_RESOLUTION_ALGORITHM_VERSION, str)
        assert "reviewed_gait_events.csv" in OUTPUT_ARTIFACT_NAMES

    def test_artifacts_dataclass(self) -> None:
        import dataclasses

        fields = {f.name for f in dataclasses.fields(ReviewResolutionArtifacts)}
        assert "artifact_directory" in fields
        assert "reviewed_gait_events_path" in fields
        assert "reviewed_strides_path" in fields
        assert "review_resolution_metadata_path" in fields
