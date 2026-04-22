"""Diagnostic tool to understand why specific videos produce 0 predictions.

For each provided video directory (containing a Labels-ball.json and an mp4),
it reports:

1.  Video metadata (fps, decoded frame count, metadata frame count, duration).
2.  Whether `_frame_nr_to_annotation_dict` would collide (the silent-crash
    scenario for 29 fps videos).
3.  Whether the extracted frames folder exists and whether the frame count
    matches the decoded frame count.
4.  The raw per-class score distribution after running inference on the whole
    video (no threshold, no SNMS) — this tells you whether the model is truly
    silent (all zeros) or just below your confidence threshold.

Usage:
    uv run python -u dudek/scripts/debug_video.py \
        --model_checkpoint_path=/workspace/bas/bt-soccer-bas2/tdeed_competition_basinit_1.pt \
        --resolution=224 \
        --clip_frames_count=170 \
        --overlap=136 \
        --val_batch_size=2 \
        --video_dir=/workspace/bas/data/competition_videos_val/my_league/2026-2027/218725_elastic_transcoded_1775659348/ \
        --video_dir=/workspace/bas/data/competition_videos_val/my_league/2026-2027/217322_elastic_transcoded_1771424285/
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import List, Tuple

import click
import numpy as np
import torch

from dudek.data.competition import Action
from dudek.data.team_bas import SoccerVideo
from dudek.ml.data.tdeed import (
    TeamTDeed2HeadsPrediction,
    TeamTDeedDataset,
)
from dudek.ml.model.tdeed.eval.two_heads import (
    BASTeamTDeedEvaluator,
    _TeamBASScoredVideo,
)
from dudek.ml.model.tdeed.modules.tdeed import TDeedModule


def _build_video(video_dir: str, resolution: int) -> SoccerVideo:
    video_dir = video_dir.rstrip("/")
    match_label = os.path.basename(video_dir)
    season = os.path.basename(os.path.dirname(video_dir))
    league = os.path.basename(os.path.dirname(os.path.dirname(video_dir)))
    root = os.path.dirname(os.path.dirname(os.path.dirname(video_dir)))
    return SoccerVideo.bas_video_from_path(
        os.path.join(root, league, season, match_label),
        resolution,
        labels_enum=Action,
    )


def _check_metadata(video: SoccerVideo) -> dict:
    metadata_fps = video.metadata_fps
    metadata_n_frames = int(video.metadata_n_frames)
    actual_n_frames = video.actual_n_frames
    actual_fps = video.actual_fps
    duration_s = metadata_n_frames / metadata_fps if metadata_fps > 0 else 0.0
    return {
        "metadata_fps": metadata_fps,
        "metadata_n_frames": metadata_n_frames,
        "actual_n_frames": actual_n_frames,
        "actual_fps": actual_fps,
        "duration_s": duration_s,
        "fps_drift_pct": (
            abs(actual_fps - metadata_fps) / metadata_fps * 100.0
            if metadata_fps > 0
            else 0.0
        ),
    }


def _check_frame_collision(video: SoccerVideo) -> dict:
    if not video.annotations:
        return {"has_annotations": False}
    frame_to_labels: dict = defaultdict(list)
    for ann in video.annotations:
        frame_nr = ann.get_frame_nr(fps=video.metadata_fps)
        frame_to_labels[frame_nr].append(
            (ann.position, getattr(ann.label, "value", str(ann.label)))
        )
    collisions = {fn: lbls for fn, lbls in frame_to_labels.items() if len(lbls) > 1}
    return {
        "has_annotations": True,
        "n_annotations": len(video.annotations),
        "n_unique_frames": len(frame_to_labels),
        "n_collisions": len(collisions),
        "collision_examples": list(collisions.items())[:5],
    }


def _check_extracted_frames(video: SoccerVideo) -> dict:
    path = video.frames_path
    if not os.path.exists(path):
        return {"exists": False, "path": path}
    frame_files = [f for f in os.listdir(path) if f.endswith(".jpg") or f.endswith(".png")]
    frame_indices = []
    for f in frame_files:
        try:
            frame_indices.append(int(f.split(".")[0]))
        except ValueError:
            pass
    return {
        "exists": True,
        "path": path,
        "n_extracted_frames": len(frame_files),
        "min_index": min(frame_indices) if frame_indices else None,
        "max_index": max(frame_indices) if frame_indices else None,
        "consecutive_gaps": (
            max(frame_indices) - min(frame_indices) + 1 - len(frame_indices)
            if frame_indices
            else 0
        ),
    }


def _run_inference(
    video: SoccerVideo,
    model: TDeedModule,
    clip_frames_count: int,
    overlap: int,
    batch_size: int,
    displacement: int = 4,
) -> np.ndarray:
    """Runs full inference on a single video, returns raw scores matrix
    [n_frames, n_action_classes] (no background, no SNMS, no threshold)."""
    clips_split = []
    for clip in video.get_clips(accepted_gap=2):
        clips_split += clip.split(
            clip_frames_count=clip_frames_count, overlap=overlap
        )
    print(f"  Generated {len(clips_split)} sub-clips of {clip_frames_count} frames each")
    if not clips_split:
        return np.zeros((0, len(Action)), dtype=np.float32)

    dataset = TeamTDeedDataset(
        clips_split,
        labels_enum=Action,
        displacement=displacement,
        return_dict=False,
        flip_proba=0.0,
        camera_move_proba=0.0,
        crop_proba=0.0,
        even_choice_proba=0.0,
        evaluate=True,
    )
    evaluator = BASTeamTDeedEvaluator(
        model=model, dataset=dataset, delta_frames_tolerance=1
    )
    predictions: List[TeamTDeed2HeadsPrediction] = []
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=lambda x: x, shuffle=False
    )
    from tqdm import tqdm as _tqdm
    for batch_of_clips in _tqdm(loader, desc="  predicting"):
        predictions += evaluator.predict(batch_of_clips)

    scored = _TeamBASScoredVideo.from_predictions(
        video, predictions, labels_enum=Action, use_snms=False
    )
    return scored.scores


def _summarise_scores(scores: np.ndarray) -> None:
    if scores.size == 0:
        print("  ** scores matrix is EMPTY - no sub-clips were produced **")
        return
    print(f"  Scores matrix shape: {scores.shape}")
    print(f"  Overall  max={scores.max():.4f}  min={scores.min():.4f}  mean={scores.mean():.4f}")
    n_nonzero = int((scores > 0).sum())
    total = scores.size
    print(f"  Non-zero cells: {n_nonzero}/{total} ({100*n_nonzero/total:.1f}%)")
    print("\n  Per-class confidence distribution (one class per row):")
    print(
        f"    {'Class':<22} {'max':>8} {'p99':>8} {'p95':>8} {'p90':>8} {'p50':>8} {'mean':>8}"
    )
    action_list = list(Action)
    for i, action in enumerate(action_list):
        col = scores[:, i]
        if col.size == 0:
            continue
        print(
            f"    {action.name:<22} "
            f"{col.max():>8.4f} "
            f"{np.percentile(col, 99):>8.4f} "
            f"{np.percentile(col, 95):>8.4f} "
            f"{np.percentile(col, 90):>8.4f} "
            f"{np.percentile(col, 50):>8.4f} "
            f"{col.mean():>8.4f}"
        )
    print("\n  Frame-level argmax distribution (ignoring background):")
    argmax = np.argmax(scores, axis=1)
    maxval = np.max(scores, axis=1)
    for thr in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
        n_above = int((maxval > thr).sum())
        print(f"    frames with max_action_score > {thr:.2f}: {n_above}")


@click.command()
@click.option("--model_checkpoint_path", required=True)
@click.option("--resolution", type=int, default=224)
@click.option("--clip_frames_count", type=int, default=170)
@click.option("--overlap", type=int, default=136)
@click.option("--val_batch_size", type=int, default=2)
@click.option("--displacement", type=int, default=4)
@click.option(
    "--video_dir",
    "video_dirs",
    multiple=True,
    required=True,
    help="Full path to a single video directory (containing Labels-ball.json and the mp4). Pass multiple times.",
)
def main(
    model_checkpoint_path: str,
    resolution: int,
    clip_frames_count: int,
    overlap: int,
    val_batch_size: int,
    displacement: int,
    video_dirs: Tuple[str, ...],
) -> None:
    print(f"Loading model from {model_checkpoint_path}")
    model = TDeedModule(
        clip_len=clip_frames_count,
        n_layers=2,
        sgp_ks=5,
        sgp_k=4,
        num_classes=len(Action),
        features_model_name="regnety_008",
        temporal_shift_mode="gsf",
        gaussian_blur_ks=5,
    )
    model.load_all(model_weight_path=model_checkpoint_path)
    model = model.cuda().eval()

    for video_dir in video_dirs:
        print("\n" + "=" * 80)
        print(f"VIDEO: {video_dir}")
        print("=" * 80)
        video = _build_video(video_dir, resolution)

        print("\n[1/4] Video metadata")
        meta = _check_metadata(video)
        for k, v in meta.items():
            print(f"    {k}: {v}")

        print("\n[2/4] Frame-collision check (ValueError scenario)")
        coll = _check_frame_collision(video)
        for k, v in coll.items():
            print(f"    {k}: {v}")

        print("\n[3/4] Extracted-frames check")
        frames = _check_extracted_frames(video)
        for k, v in frames.items():
            print(f"    {k}: {v}")

        if not frames.get("exists", False):
            print("    ** Skipping inference: frames not extracted **")
            continue

        print("\n[4/4] Raw inference (no SNMS, no threshold)")
        with torch.inference_mode():
            scores = _run_inference(
                video, model, clip_frames_count, overlap, val_batch_size, displacement
            )
        _summarise_scores(scores)


if __name__ == "__main__":
    main()
