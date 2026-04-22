"""Count events per class across one or more dataset paths.

Walks a directory tree <root>/<league>/<season>/<match_label>/Labels-ball.json
and prints per-class event counts for each dataset path, plus a comparison
table (train vs val absolute counts and train share %).

Usage:
    uv run python -u dudek/scripts/count_events.py \
        --path=/workspace/bas/data/competition_videos/ \
        --path=/workspace/bas/data/competition_videos_val/
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import click

from dudek.utils.competition_score import COMPETITION_ACTIONS


def _find_label_files(root: str) -> List[str]:
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        if "Labels-ball.json" in filenames:
            out.append(os.path.join(dirpath, "Labels-ball.json"))
    return out


def _count_events_in_file(
    path: str,
) -> Tuple[Dict[str, int], int, int]:
    """Returns (per_class_count, total_events, duration_ms_guess)."""
    per_class: Dict[str, int] = defaultdict(int)
    total = 0
    max_pos_ms = 0
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"WARN could not parse {path}: {e}")
        return per_class, 0, 0

    for ann in data.get("annotations", []):
        label = str(ann.get("label", "")).lower()
        try:
            pos_ms = int(ann.get("position", 0))
        except (TypeError, ValueError):
            pos_ms = 0
        max_pos_ms = max(max_pos_ms, pos_ms)
        per_class[label] += 1
        total += 1
    return per_class, total, max_pos_ms


def _summarise_dataset(root: str) -> Dict[str, object]:
    files = _find_label_files(root)
    per_class_total: Dict[str, int] = defaultdict(int)
    total_events = 0
    total_duration_ms = 0
    per_video: List[Tuple[str, int, int]] = []

    for f in files:
        counts, total, dur_ms = _count_events_in_file(f)
        for k, v in counts.items():
            per_class_total[k] += v
        total_events += total
        total_duration_ms += dur_ms
        rel = os.path.relpath(os.path.dirname(f), root)
        per_video.append((rel, total, dur_ms))

    return {
        "root": root,
        "n_videos": len(files),
        "total_events": total_events,
        "total_duration_hours": total_duration_ms / 1000 / 3600,
        "per_class": dict(per_class_total),
        "per_video": per_video,
    }


def _print_dataset(summary: Dict[str, object]) -> None:
    print(f"\n=== {summary['root']}")
    print(f"Videos: {summary['n_videos']}")
    print(f"Total events: {summary['total_events']}")
    print(f"Est. total duration (last-event-position): {summary['total_duration_hours']:.2f} h")
    pc = summary["per_class"]
    print(f"\n  {'Class':<22} {'Count':>8} {'Weight':>8}")
    print("  " + "-" * 42)
    for cls in COMPETITION_ACTIONS.keys():
        w = COMPETITION_ACTIONS[cls][0]
        print(f"  {cls:<22} {pc.get(cls, 0):>8} {w:>8.1f}")
    non_standard = sorted(set(pc.keys()) - set(COMPETITION_ACTIONS.keys()))
    if non_standard:
        print(f"\n  Non-standard labels in this dataset:")
        for cls in non_standard:
            print(f"    {cls:<22} {pc[cls]:>8}")


def _print_comparison(summaries: List[Dict[str, object]]) -> None:
    if len(summaries) < 2:
        return
    print("\n" + "=" * 80)
    print("Cross-dataset comparison (event counts per class)")
    print("=" * 80)

    headers = ["Class", "Weight"] + [os.path.basename(s["root"].rstrip("/")) for s in summaries] + ["Share in 1st"]
    row_fmt = "  {:<22} {:>7} " + " ".join("{:>14}" for _ in summaries) + " {:>14}"
    print(row_fmt.format(*headers))
    print("  " + "-" * (22 + 8 + 15 * len(summaries) + 15))

    for cls in COMPETITION_ACTIONS.keys():
        w = COMPETITION_ACTIONS[cls][0]
        counts = [s["per_class"].get(cls, 0) for s in summaries]
        total = sum(counts)
        share1 = (counts[0] / total * 100.0) if total else 0.0
        row = [cls, f"{w:.1f}"] + [str(c) for c in counts] + [f"{share1:.1f}%"]
        print(row_fmt.format(*row))


@click.command()
@click.option(
    "--path",
    "paths",
    multiple=True,
    required=True,
    help="Dataset root path. Pass multiple --path flags to compare (e.g. train vs val).",
)
def main(paths: Tuple[str, ...]) -> None:
    summaries = [_summarise_dataset(p) for p in paths]
    for s in summaries:
        _print_dataset(s)
    _print_comparison(summaries)


if __name__ == "__main__":
    main()
