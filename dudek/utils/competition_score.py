"""Competition scoring utilities.

Mirrors the scoring logic used by the competition (see bt-manual-bas/manual-bas-v2/score.py).
For each GT event, matches the closest unmatched prediction of the SAME CLASS within the
class-specific tolerance; scores +weight * (1 - diff/tol) for matches and -weight for
unmatched predictions (FPs).

Final per-video score:
    raw   = (matched_score - fp_penalty) / total_gt_weight
    final = clamp(raw, 0, 1)

The 15-class official ACTIONS list is used to compute `total_gt_weight` (denominator),
regardless of which classes we choose to submit. GT events of classes we don't submit
become unmatched and contribute 0 to the numerator (but still weigh the denominator).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Optional

# Official 15-class competition action list: label -> (weight, tolerance_seconds).
# Note: the scoring server is the source of truth for spellings. "aerial_duel" is the
# correct spelling per the user; the reference score.py had a typo ("arial_duel") which
# we ignore here.
COMPETITION_ACTIONS: dict[str, tuple[float, float]] = {
    "pass": (1.0, 1.0),
    "pass_received": (1.4, 1.0),
    "recovery": (1.5, 1.5),
    "tackle": (2.5, 1.5),
    "interception": (2.8, 2.0),
    "ball_out_of_play": (2.9, 2.0),
    "clearance": (3.1, 2.0),
    "take_on": (3.2, 2.0),
    "substitution": (4.2, 2.0),
    "block": (4.2, 2.0),
    "aerial_duel": (4.3, 2.0),
    "shot": (4.7, 2.0),
    "save": (7.3, 2.0),
    "foul": (7.7, 2.5),
    "goal": (10.9, 3.0),
}

MIN_SCORE_FLOOR = 0.0  # same default as score.py


@dataclass(frozen=True)
class Event:
    """An event (ground-truth or predicted)."""

    label: str
    time_ms: int
    # frame_nr and confidence are optional; only used for reporting/debugging.
    frame_nr: Optional[int] = None
    confidence: Optional[float] = None


@dataclass
class VideoScoreRow:
    """One row of the match table for a scored video (mirrors score.py rows)."""

    kind: str  # "MATCH", "MISS", or "FP"
    label: str
    gt_frame: Optional[int]
    gt_ms: Optional[int]
    pred_frame: Optional[int]
    pred_ms: Optional[int]
    diff_ms: Optional[int]
    points: float


@dataclass
class VideoScoreResult:
    matched_score: float
    fp_penalty: float
    total_gt_weight: float
    raw_score: float
    final_score: float  # clamped to [0, 1]
    rows: list[VideoScoreRow]
    # per-class decomposition: {label -> (matched_points, fp_penalty, n_matched, n_fp, n_gt)}
    per_class: dict[str, dict[str, float]]


def load_all_gt_events(labels_path: str) -> list[Event]:
    """Load all events from a Labels-ball.json file, across ALL 15 competition classes.

    Unknown labels (not in COMPETITION_ACTIONS) are dropped silently — consistent with
    score.py behaviour (and the scoring server).
    """
    if not os.path.exists(labels_path):
        return []
    with open(labels_path, "r") as f:
        data = json.load(f)

    events: list[Event] = []
    for ann in data.get("annotations", []):
        label = ann.get("label")
        if label not in COMPETITION_ACTIONS:
            continue
        try:
            time_ms = int(ann.get("position"))
        except (TypeError, ValueError):
            continue
        events.append(Event(label=label, time_ms=time_ms))
    return events


def score_video(
    gt_events: Iterable[Event],
    pred_events: Iterable[Event],
    actions: Optional[dict[str, tuple[float, float]]] = None,
    min_score_floor: float = MIN_SCORE_FLOOR,
) -> VideoScoreResult:
    """Score one video's predictions against ground truth.

    Matches score.py logic exactly:
      - iterate GT in list order
      - for each GT, pick the closest unmatched prediction of the SAME class within tolerance
      - decay = 1 - (diff / tol_ms) * (1 - min_score_floor) [linear]
      - matched points = weight * decay
      - unmatched predictions are FPs with penalty -weight
      - total_gt_weight is over ALL 15 classes present in gt_events (even unsubmitted)

    Args:
        gt_events: iterable of ground-truth Event objects. MUST include events of
            classes we don't predict (e.g. clearance) so the denominator is correct.
        pred_events: iterable of predicted Event objects. Only classes we choose to
            submit should be here.
        actions: label -> (weight, tolerance_seconds). Defaults to COMPETITION_ACTIONS.
        min_score_floor: minimum decay at tolerance edge. 0 = full decay (score.py default).

    Returns:
        VideoScoreResult with per-class decomposition and clamped final score.
    """
    actions = actions or COMPETITION_ACTIONS
    gt_list = [e for e in gt_events if e.label in actions]
    pred_list = list(pred_events)

    # Sanity: remove predictions with unknown labels; they are silently dropped by
    # the scoring server anyway.
    pred_list = [p for p in pred_list if p.label in actions]

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    rows: list[VideoScoreRow] = []
    per_class: dict[str, dict[str, float]] = {}

    def _bump(label: str, field: str, val: float = 1.0):
        d = per_class.setdefault(
            label,
            {"matched_points": 0.0, "fp_penalty": 0.0, "n_matched": 0, "n_fp": 0, "n_gt": 0},
        )
        d[field] += val

    # Count GT per class
    for gt in gt_list:
        _bump(gt.label, "n_gt")

    for i, gt in enumerate(gt_list):
        weight, tol_s = actions[gt.label]
        tol_ms = tol_s * 1000.0
        best_j, best_diff = None, float("inf")
        for j, pr in enumerate(pred_list):
            if j in matched_pred:
                continue
            if pr.label != gt.label:
                continue
            diff = abs(pr.time_ms - gt.time_ms)
            if diff <= tol_ms and diff < best_diff:
                best_j, best_diff = j, diff
        if best_j is not None:
            matched_pred.add(best_j)
            matched_gt.add(i)
            decay = 1.0 - (best_diff / tol_ms) * (1.0 - min_score_floor)
            pts = weight * decay
            pr = pred_list[best_j]
            rows.append(
                VideoScoreRow(
                    kind="MATCH",
                    label=gt.label,
                    gt_frame=gt.frame_nr,
                    gt_ms=gt.time_ms,
                    pred_frame=pr.frame_nr,
                    pred_ms=pr.time_ms,
                    diff_ms=int(best_diff),
                    points=pts,
                )
            )
            _bump(gt.label, "matched_points", pts)
            _bump(gt.label, "n_matched")
        else:
            rows.append(
                VideoScoreRow(
                    kind="MISS",
                    label=gt.label,
                    gt_frame=gt.frame_nr,
                    gt_ms=gt.time_ms,
                    pred_frame=None,
                    pred_ms=None,
                    diff_ms=None,
                    points=0.0,
                )
            )

    for j, pr in enumerate(pred_list):
        if j in matched_pred:
            continue
        weight, _ = actions[pr.label]
        rows.append(
            VideoScoreRow(
                kind="FP",
                label=pr.label,
                gt_frame=None,
                gt_ms=None,
                pred_frame=pr.frame_nr,
                pred_ms=pr.time_ms,
                diff_ms=None,
                points=-weight,
            )
        )
        _bump(pr.label, "fp_penalty", weight)
        _bump(pr.label, "n_fp")

    total_gt_weight = sum(actions[gt.label][0] for gt in gt_list)
    matched_score = sum(r.points for r in rows if r.kind == "MATCH")
    fp_penalty = sum(-r.points for r in rows if r.kind == "FP")
    raw_score = (matched_score - fp_penalty) / total_gt_weight if total_gt_weight > 0 else 0.0
    final_score = max(0.0, min(1.0, raw_score))

    return VideoScoreResult(
        matched_score=matched_score,
        fp_penalty=fp_penalty,
        total_gt_weight=total_gt_weight,
        raw_score=raw_score,
        final_score=final_score,
        rows=rows,
        per_class=per_class,
    )


def score_class_on_video(
    gt_events: Iterable[Event],
    pred_events_for_class: Iterable[Event],
    class_label: str,
    actions: Optional[dict[str, tuple[float, float]]] = None,
    min_score_floor: float = MIN_SCORE_FLOOR,
) -> tuple[float, float, int, int, int]:
    """Compute a single class's (matched_points, fp_penalty, n_matched, n_fp, n_gt) on one video.

    GT events of classes OTHER than class_label are ignored (they don't affect this class's
    contribution under the same-class matching rule). GT count / weight is still needed
    for the global denominator, but this helper is intended for per-class optimization
    where we compare contributions across (threshold, snms_window) settings.

    The predictions passed in MUST all be of class `class_label`.
    """
    actions = actions or COMPETITION_ACTIONS
    if class_label not in actions:
        return 0.0, 0.0, 0, 0, 0

    weight, tol_s = actions[class_label]
    tol_ms = tol_s * 1000.0

    gt_of_class = [e for e in gt_events if e.label == class_label]
    pred_of_class = [p for p in pred_events_for_class if p.label == class_label]

    matched_pred: set[int] = set()
    matched_points = 0.0
    n_matched = 0

    for gt in gt_of_class:
        best_j, best_diff = None, float("inf")
        for j, pr in enumerate(pred_of_class):
            if j in matched_pred:
                continue
            diff = abs(pr.time_ms - gt.time_ms)
            if diff <= tol_ms and diff < best_diff:
                best_j, best_diff = j, diff
        if best_j is not None:
            matched_pred.add(best_j)
            decay = 1.0 - (best_diff / tol_ms) * (1.0 - min_score_floor)
            matched_points += weight * decay
            n_matched += 1

    n_fp = len(pred_of_class) - len(matched_pred)
    fp_penalty = weight * n_fp
    return matched_points, fp_penalty, n_matched, n_fp, len(gt_of_class)


def aggregate_final_scores(per_video_results: list[VideoScoreResult]) -> dict:
    """Summarise per-video scores the way the competition does: average of clamped finals."""
    if not per_video_results:
        return {
            "n_videos": 0,
            "avg_final_score": 0.0,
            "avg_raw_score": 0.0,
            "sum_matched": 0.0,
            "sum_fp_penalty": 0.0,
            "sum_gt_weight": 0.0,
        }
    n = len(per_video_results)
    return {
        "n_videos": n,
        "avg_final_score": sum(r.final_score for r in per_video_results) / n,
        "avg_raw_score": sum(r.raw_score for r in per_video_results) / n,
        "sum_matched": sum(r.matched_score for r in per_video_results),
        "sum_fp_penalty": sum(r.fp_penalty for r in per_video_results),
        "sum_gt_weight": sum(r.total_gt_weight for r in per_video_results),
    }
