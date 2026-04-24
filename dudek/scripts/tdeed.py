import enum
import json
import math
import os
from collections import Counter
from typing import List, Tuple, Type

import numpy as np
import torch
import wandb
from tqdm import tqdm

from dudek.config import (
    EXPERIMENTS_RANDOM_SEED,
    DEFAULT_DEVICE,
    TEST_SET_CHALLENGE_SEED,
)
from dudek.data.team_bas import ActionLabel, BASLabel
from dudek.data.competition import Action, ACTION_CONFIGS
from dudek.ml.data.tdeed import TeamTDeedDataset
from dudek.ml.model.tdeed.eval.base import TDeedMAPEvaluator
from dudek.ml.model.tdeed.eval.two_heads import BASTeamTDeedEvaluator
from dudek.ml.model.tdeed.modules.tdeed import TDeedModule


from dudek.ml.model.tdeed.training.two_heads import train as train_tdeed
from dudek.utils.common import soft_non_maximum_suppression
from dudek.utils.competition_score import (
    COMPETITION_ACTIONS,
    Event,
    aggregate_final_scores,
    load_all_gt_events,
    score_class_on_video,
    score_video,
)
from dudek.utils.video import load_action_spotting_videos, load_bas_videos, load_competition_videos

import click

cli = click.Group()


def compute_inverse_sqrt_class_weights(
    videos,
    labels_enum: Type[enum.Enum],
    cap: float = 3.0,
) -> Tuple[List[float], List[int]]:
    """
    Inverse-sqrt class weights from annotation counts across videos.

    For class ``c`` with count ``n_c`` and most-common count ``n_max``:
        w_c = min(sqrt(n_max / n_c), cap)

    Returns (weights_in_enum_order, counts_in_enum_order). Weights are
    multiplicative relative factors; downstream loss still multiplies by
    ``foreground_weight``.
    """
    counter: Counter = Counter()
    for v in videos:
        for ann in getattr(v, "annotations", None) or []:
            counter[ann.label] += 1
    class_list = list(labels_enum)
    counts = [counter.get(c, 0) for c in class_list]
    n_max = max(counts) if counts else 1
    if n_max == 0:
        n_max = 1
    weights: List[float] = []
    for n in counts:
        if n <= 0:
            weights.append(float(cap))
        else:
            weights.append(float(min(math.sqrt(n_max / n), cap)))
    return weights, counts


@cli.command()
@click.option("--dataset_path", type=str, required=True)
@click.option("--resolution", type=int, default=224)
@click.option("--clip_frames_count", type=int, default=70)
@click.option("--overlap", type=int, default=55)
@click.option("--displacement", type=int, default=4)
@click.option("--flip_proba", type=float, default=0.1)
@click.option("--camera_move_proba", type=float, default=0.1)
@click.option("--crop_proba", type=float, default=0.1)
@click.option("--even_choice_proba", type=float, default=0.0)
@click.option("--nr_epochs", type=int, default=40)
@click.option("--warm_up_epochs", type=int, default=1)
@click.option("--learning_rate", type=float, default=0.0006)
@click.option("--train_batch_size", type=int, default=8)
@click.option("--val_batch_size", type=int, default=8)
@click.option("--loss_foreground_weight", type=int, default=5)
@click.option("--eval_metric", type=str, default="loss")
@click.option("--start_eval_epoch_nr", type=int, default=0)
@click.option("--features_model_name", type=str, default="regnety_008")
@click.option("--temporal_shift_mode", type=str, default="gsf")
@click.option("--acc_grad_iter", type=int, default=1)
@click.option("--enforce_train_epoch_size", type=int, default=None)
@click.option("--enforce_val_epoch_size", type=int, default=None)
@click.option("--gaussian_blur_kernel_size", type=int, default=5)
@click.option("--tdeed_arch_n_layers", type=int, default=2)
@click.option("--tdeed_arch_sgp_ks", type=int, default=5)
@click.option("--tdeed_arch_sgp_k", type=int, default=4)
@click.option("--save_as", type=str, default="tdeed_pretrained.pt")
@click.option("--model_checkpoint_path", type=str, default=None)
@click.option("--experiment_name", type=str, default="tdeed_pretraining")
@click.option("--random_seed", type=int, default=EXPERIMENTS_RANDOM_SEED)
@click.option("--use_wandb", type=bool, default=False)

def pretrain(
    dataset_path: str,
    resolution: int = 224,
    clip_frames_count: int = 80,
    overlap: int = 60,
    displacement: int = 4,
    flip_proba: float = 0.1,
    camera_move_proba: float = 0.1,
    crop_proba: float = 0.1,
    even_choice_proba: float = 0.0,
    nr_epochs: int = 25,
    warm_up_epochs: int = 1,
    learning_rate: float = 0.0006,
    train_batch_size: int = 8,
    val_batch_size: int = 8,
    loss_foreground_weight: int = 5,  # how to weight event class in loss function <= background weight is 1,
    eval_metric="loss",
    start_eval_epoch_nr: int = 0,
    features_model_name: str = "regnety_008",
    temporal_shift_mode: str = "gsf",
    acc_grad_iter: int = 1,
    enforce_train_epoch_size: int = None,
    enforce_val_epoch_size: int = None,
    gaussian_blur_kernel_size: int = 5,
    tdeed_arch_n_layers: int = 2,
    tdeed_arch_sgp_ks: int = 5,
    tdeed_arch_sgp_k: int = 4,
    save_as: str = "tdeed_pretrained.pt",
    model_checkpoint_path: str = None,
    experiment_name: str = "tdeed_pretraining",
    random_seed: int = EXPERIMENTS_RANDOM_SEED,
    use_wandb: bool = False
):
    assert resolution in [224, 720]

    if use_wandb:
        wandb.init(project=experiment_name, sync_tensorboard=True)

    videos = load_action_spotting_videos(
        dataset_path, resolution=resolution, random_team_when_no_team=True
    )
    all_clips = []
    for v in tqdm(videos, desc="Loading clips ...", total=len(videos)):
        clips = v.get_clips(
            accepted_gap=2,
        )
        for clip in clips:
            all_clips += clip.split(
                clip_frames_count=clip_frames_count, overlap=overlap
            )

    all_dataset = TeamTDeedDataset(
        all_clips,
        labels_enum=ActionLabel,
        displacement=displacement,
        return_dict=True,
        flip_proba=flip_proba,
        camera_move_proba=camera_move_proba,
        crop_proba=crop_proba,
        even_choice_proba=even_choice_proba,
    )

    n_matches = len(all_dataset.get_unique_matches())
    assert n_matches >= 2, f"Need at least 2 unique matches, got {n_matches}"
    n_val = max(1, round(n_matches * 0.08))   # ~8 % for validation, at least 1
    n_train = n_matches - n_val
    print(f"Splitting {n_matches} matches → train={n_train}, val={n_val}")
    train_dataset, val_dataset = all_dataset.split_by_matches(
        counts=[n_train, n_val], random_seed=random_seed
    )

    if enforce_train_epoch_size is not None:
        train_dataset.enforced_epoch_size = enforce_train_epoch_size

    if eval_metric == "map":
        val_dataset.return_dict = False
    val_dataset.flip_proba = 0.0
    val_dataset.camera_move_proba = 0.0
    val_dataset.crop_proba = 0.0
    val_dataset.even_choice_proba = 0.0

    if enforce_val_epoch_size is not None:
        val_dataset.enforced_epoch_size = enforce_val_epoch_size

    tdeed_model = TDeedModule(
        clip_len=clip_frames_count,
        n_layers=tdeed_arch_n_layers,
        sgp_ks=tdeed_arch_sgp_ks,
        sgp_k=tdeed_arch_sgp_k,
        num_classes=len(ActionLabel),
        features_model_name=features_model_name,
        temporal_shift_mode=temporal_shift_mode,
        gaussian_blur_ks=gaussian_blur_kernel_size,
    )

    if model_checkpoint_path:
        tdeed_model.load_all(model_checkpoint_path)

    train_tdeed(
        experiment_name=experiment_name,
        model=tdeed_model,
        labels_enum=ActionLabel,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        eval_metric=eval_metric,
        nr_epochs=nr_epochs,
        start_eval_epoch_nr=start_eval_epoch_nr,
        foreground_weight=loss_foreground_weight,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        device=DEFAULT_DEVICE,
        acc_grad_iter=acc_grad_iter,
        warm_up_epochs=warm_up_epochs,
        save_as=save_as,
        lr=learning_rate,
    )


@cli.command()
@click.option("--dataset_path", type=str, required=True)
@click.option("--resolution", type=int, default=224)
@click.option("--clip_frames_count", type=int, default=70)
@click.option("--overlap", type=int, default=55)
@click.option("--displacement", type=int, default=4)
@click.option("--flip_proba", type=float, default=0.1)
@click.option("--camera_move_proba", type=float, default=0.1)
@click.option("--crop_proba", type=float, default=0.1)
@click.option("--even_choice_proba", type=float, default=0.0)
@click.option("--nr_epochs", type=int, default=40)
@click.option("--warm_up_epochs", type=int, default=1)
@click.option("--learning_rate", type=float, default=0.0006)
@click.option("--train_batch_size", type=int, default=8)
@click.option("--val_batch_size", type=int, default=8)
@click.option("--eval_metric", type=str, default="map")
@click.option("--start_eval_epoch_nr", type=int, default=0)
@click.option("--loss_foreground_weight", type=int, default=5)
@click.option("--features_model_name", type=str, default="regnety_008")
@click.option("--temporal_shift_mode", type=str, default="gsf")
@click.option("--acc_grad_iter", type=int, default=1)
@click.option("--enforce_train_epoch_size", type=int, default=None)
@click.option("--enforce_val_epoch_size", type=int, default=None)
@click.option("--gaussian_blur_kernel_size", type=int, default=5)
@click.option("--tdeed_arch_n_layers", type=int, default=2)
@click.option("--tdeed_arch_sgp_ks", type=int, default=5)
@click.option("--tdeed_arch_sgp_k", type=int, default=4)
@click.option("--model_checkpoint_path", type=str, default=None)
@click.option("--save_as", type=str, default="tdeed_best.pt")
@click.option("--experiment_name", type=str, default="tdeed_training")
@click.option("--random_seed", type=int, default=TEST_SET_CHALLENGE_SEED)
@click.option("--use_wandb", type=bool, default=False)
def train(
    dataset_path: str,
    resolution: int = 224,
    clip_frames_count: int = 80,
    overlap: int = 60,
    displacement: int = 4,
    flip_proba: float = 0.2,
    camera_move_proba: float = 0.2,
    crop_proba: float = 0.2,
    even_choice_proba: float = 0.0,
    nr_epochs: int = 25,
    warm_up_epochs: int = 1,
    learning_rate: float = 0.0006,
    train_batch_size: int = 8,
    val_batch_size: int = 8,
    eval_metric="map",
    start_eval_epoch_nr: int = 0,
    loss_foreground_weight: int = 5,  # how to weight event class in loss function <= background weight is 1,
    features_model_name: str = "regnety_008",
    temporal_shift_mode: str = "gsf",
    acc_grad_iter: int = 1,
    enforce_train_epoch_size: int = None,
    enforce_val_epoch_size: int = None,
    gaussian_blur_kernel_size: int = 5,
    tdeed_arch_n_layers: int = 2,
    tdeed_arch_sgp_ks: int = 5,
    tdeed_arch_sgp_k: int = 4,
    model_checkpoint_path: str = None,
    save_as: str = "tdeed_best.pt",
    experiment_name: str = "tdeed_training",
    random_seed: int = TEST_SET_CHALLENGE_SEED,
    use_wandb: bool = False,
):
    assert resolution in [224, 720]
    if use_wandb:
        wandb.init(project=experiment_name, sync_tensorboard=True)


    videos = load_bas_videos(dataset_path, resolution=resolution)
    all_clips = []
    for v in tqdm(videos, desc="Loading clips ..."):
        clips = v.get_clips(
            accepted_gap=2,
        )
        for clip in clips:
            short_clips = clip.split(
                clip_frames_count=clip_frames_count, overlap=overlap
            )
            all_clips += short_clips

    all_dataset = TeamTDeedDataset(
        all_clips,
        labels_enum=BASLabel,
        displacement=displacement,
        return_dict=True,
        flip_proba=flip_proba,
        camera_move_proba=camera_move_proba,
        crop_proba=crop_proba,
        even_choice_proba=even_choice_proba,
    )

    train_dataset, val_dataset, test_dataset = all_dataset.split_by_matches(
        counts=[4, 1, 2], random_seed=random_seed
    )

    if enforce_train_epoch_size is not None:
        train_dataset.enforced_epoch_size = enforce_train_epoch_size

    if eval_metric == "map":
        val_dataset.return_dict = False

    val_dataset.flip_proba = 0.0
    val_dataset.camera_move_proba = 0.0
    val_dataset.crop_proba = 0.0
    val_dataset.even_choice_proba = 0.0

    if enforce_val_epoch_size is not None:
        val_dataset.enforced_epoch_size = enforce_val_epoch_size

    tdeed_model = TDeedModule(
        clip_len=clip_frames_count,
        n_layers=tdeed_arch_n_layers,
        sgp_ks=tdeed_arch_sgp_ks,
        sgp_k=tdeed_arch_sgp_k,
        num_classes=len(BASLabel),
        features_model_name=features_model_name,
        temporal_shift_mode=temporal_shift_mode,
        gaussian_blur_ks=gaussian_blur_kernel_size,
    )
    if model_checkpoint_path:
        tdeed_model.load_backbone(model_weight_path=model_checkpoint_path)

    train_tdeed(
        experiment_name=experiment_name,
        model=tdeed_model,
        labels_enum=BASLabel,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        eval_metric=eval_metric,
        nr_epochs=nr_epochs,
        start_eval_epoch_nr=start_eval_epoch_nr,
        foreground_weight=loss_foreground_weight,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        device=DEFAULT_DEVICE,
        acc_grad_iter=acc_grad_iter,
        save_as=save_as,
        lr=learning_rate,
        loss_weights=[1.5, 1],
        warm_up_epochs=warm_up_epochs,
    )


@cli.command()
@click.option("--dataset_path", type=str, required=True)
@click.option("--resolution", type=int, default=224)
@click.option("--clip_frames_count", type=int, default=80)
@click.option("--overlap", type=int, default=68)
@click.option("--displacement", type=int, default=4)
@click.option("--flip_proba", type=float, default=0.1)
@click.option("--camera_move_proba", type=float, default=0.1)
@click.option("--crop_proba", type=float, default=0.1)
@click.option("--even_choice_proba", type=float, default=0.0)
@click.option("--nr_epochs", type=int, default=40)
@click.option("--warm_up_epochs", type=int, default=1)
@click.option("--learning_rate", type=float, default=0.0008)
@click.option("--train_batch_size", type=int, default=8)
@click.option("--loss_foreground_weight", type=int, default=5)
@click.option("--features_model_name", type=str, default="regnety_008")
@click.option("--temporal_shift_mode", type=str, default="gsf")
@click.option("--acc_grad_iter", type=int, default=1)
@click.option("--enforce_train_epoch_size", type=int, default=None)
@click.option("--gaussian_blur_kernel_size", type=int, default=5)
@click.option("--tdeed_arch_n_layers", type=int, default=2)
@click.option("--tdeed_arch_sgp_ks", type=int, default=5)
@click.option("--tdeed_arch_sgp_k", type=int, default=4)
@click.option("--save_as", type=str, default="tdeed_challenge.pt")
@click.option("--model_checkpoint_path", type=str, default="tdeed_pretrained.pt")
@click.option("--experiment_name", type=str, default=None)
@click.option("--use_wandb", type=bool, default=False)
def train_challenge(
    dataset_path: str,
    resolution: int = 224,
    clip_frames_count: int = 80,
    overlap: int = 68,
    displacement: int = 4,
    flip_proba: float = 0.1,
    camera_move_proba: float = 0.1,
    crop_proba: float = 0.1,
    even_choice_proba: float = 0.0,
    nr_epochs: int = 40,
    warm_up_epochs: int = 1,
    learning_rate: float = 0.0008,
    train_batch_size: int = 8,
    loss_foreground_weight: int = 5,  # how to weight event class in loss function <= background weight is 1,
    features_model_name: str = "regnety_008",
    temporal_shift_mode: str = "gsf",
    acc_grad_iter: int = 1,
    enforce_train_epoch_size: int = None,
    gaussian_blur_kernel_size: int = 5,
    tdeed_arch_n_layers: int = 2,
    tdeed_arch_sgp_ks: int = 5,
    tdeed_arch_sgp_k: int = 4,
    save_as: str = "tdeed_challenge.pt",
    model_checkpoint_path: str = None,
    experiment_name: str = "tdeed_training_challenge",
    use_wandb: bool = False
):
    assert resolution in [224, 720]
    if use_wandb:
        wandb.init(project=experiment_name, sync_tensorboard=True)

    videos = load_bas_videos(dataset_path, resolution=resolution)
    all_clips = []
    for v in tqdm(videos, desc="Loading clips ..."):
        clips = v.get_clips(
            accepted_gap=2,
        )
        for clip in clips:
            short_clips = clip.split(
                clip_frames_count=clip_frames_count, overlap=overlap
            )
            all_clips += short_clips

    train_dataset = TeamTDeedDataset(
        all_clips,
        labels_enum=BASLabel,
        displacement=displacement,
        return_dict=True,
        flip_proba=flip_proba,
        camera_move_proba=camera_move_proba,
        crop_proba=crop_proba,
        even_choice_proba=even_choice_proba,
    )
    if enforce_train_epoch_size is not None:
        train_dataset.enforced_epoch_size = enforce_train_epoch_size

    tdeed_model = TDeedModule(
        clip_len=clip_frames_count,
        n_layers=tdeed_arch_n_layers,
        sgp_ks=tdeed_arch_sgp_ks,
        sgp_k=tdeed_arch_sgp_k,
        num_classes=len(BASLabel),
        features_model_name=features_model_name,
        temporal_shift_mode=temporal_shift_mode,
        gaussian_blur_ks=gaussian_blur_kernel_size,
    )

    if model_checkpoint_path:
        tdeed_model.load_backbone(model_weight_path=model_checkpoint_path)

    model = train_tdeed(
        experiment_name=experiment_name,
        model=tdeed_model,
        labels_enum=BASLabel,
        train_dataset=train_dataset,
        nr_epochs=nr_epochs,
        foreground_weight=loss_foreground_weight,
        train_batch_size=train_batch_size,
        device=DEFAULT_DEVICE,
        acc_grad_iter=acc_grad_iter,
        lr=learning_rate,
        loss_weights=[1.5, 1],
        warm_up_epochs=warm_up_epochs,
        eval_metric="loss",
        val_dataset=None,
        start_eval_epoch_nr=0,
    )

    if int(os.environ.get("LOCAL_RANK", -1)) in (-1, 0):
        torch.save(
            model.state_dict(),
            save_as,
        )



@cli.command()
@click.option("--dataset_path", type=str, required=True)
@click.option("--resolution", type=int, default=224)
@click.option("--clip_frames_count", type=int, default=80)
@click.option("--overlap", type=int, default=68)
@click.option("--displacement", type=int, default=4)
@click.option("--val_batch_size", type=int, default=8)
@click.option("--features_model_name", type=str, default="regnety_008")
@click.option("--temporal_shift_mode", type=str, default="gsf")
@click.option("--t_deed_arch_n_layers", type=int, default=2)
@click.option("--t_deed_arch_sgp_ks", type=int, default=5)
@click.option("--t_deed_arch_sgp_k", type=int, default=4)
@click.option("--model_checkpoint_path", type=str, default="tdeed_challenge.pt")
@click.option("--solution_archive_file_base_name", type=str, default="solution")
def create_solution(
    dataset_path: str,
    resolution: int = 224,
    clip_frames_count: int = 80,
    overlap: int = 68,
    displacement: int = 4,
    val_batch_size: int = 8,
    features_model_name: str = "regnety_008",
    temporal_shift_mode: str = "gsf",
    t_deed_arch_n_layers: int = 2,
    t_deed_arch_sgp_ks: int = 5,
    t_deed_arch_sgp_k: int = 4,
    model_checkpoint_path: str = "tdeed_challenge.pt",
    solution_archive_file_base_name: str = "solution",
):
    assert resolution in [224, 720]
    videos = load_bas_videos(dataset_path, resolution=resolution)
    all_clips = []
    for v in tqdm(videos, desc="Loading clips ..."):
        clips = v.get_clips(
            accepted_gap=2,
        )
        for clip in clips:
            all_clips += clip.split(
                clip_frames_count=clip_frames_count, overlap=overlap
            )

    challenge_dataset = TeamTDeedDataset(
        all_clips,
        labels_enum=BASLabel,
        displacement=displacement,
        return_dict=False,
        flip_proba=0.0,
        camera_move_proba=0.0,
        crop_proba=0.0,
        even_choice_proba=0.0,
    )

    assert challenge_dataset.flip_proba == 0.0
    assert challenge_dataset.camera_move_proba == 0.0
    assert challenge_dataset.crop_proba == 0.0
    assert challenge_dataset.even_choice_proba == 0.0

    tdeed_model = TDeedModule(
        clip_len=clip_frames_count,
        n_layers=t_deed_arch_n_layers,
        sgp_ks=t_deed_arch_sgp_ks,
        sgp_k=t_deed_arch_sgp_k,
        num_classes=len(BASLabel),
        features_model_name=features_model_name,
        temporal_shift_mode=temporal_shift_mode,
    )
    tdeed_model.load_all(model_weight_path=model_checkpoint_path)
    tdeed_model = tdeed_model.to(DEFAULT_DEVICE)
    tdeed_model.eval()
    evaluator = BASTeamTDeedEvaluator(
        model=tdeed_model,
        dataset=challenge_dataset,
        delta_frames_tolerance=1,
    )

    scored_videos = evaluator.get_scored_videos(
        batch_size=val_batch_size,
        use_snms=True,
        use_hflip=False,
        snms_params=dict(class_window=12, threshold=0.01),  # empirically found
    )

    evaluator.create_solution_file(
        scored_videos=scored_videos,
        zip_output_file_name=solution_archive_file_base_name,
    )


@cli.command()
@click.option("--dataset_path", type=str, required=True)
@click.option("--resolution", type=int, default=224)
@click.option("--clip_frames_count", type=int, default=80)
@click.option("--overlap", type=int, default=68)
@click.option("--displacement", type=int, default=4)
@click.option("--flip_proba", type=float, default=0.2)
@click.option("--camera_move_proba", type=float, default=0.2)
@click.option("--crop_proba", type=float, default=0.2)
@click.option("--even_choice_proba", type=float, default=0.0)
@click.option("--nr_epochs", type=int, default=40)
@click.option("--warm_up_epochs", type=int, default=1)
@click.option("--learning_rate", type=float, default=0.0006)
@click.option("--train_batch_size", type=int, default=8)
@click.option("--val_batch_size", type=int, default=8)
@click.option("--loss_foreground_weight", type=int, default=5)
@click.option("--eval_metric", type=str, default="loss")
@click.option("--start_eval_epoch_nr", type=int, default=0)
@click.option("--features_model_name", type=str, default="regnety_008")
@click.option("--temporal_shift_mode", type=str, default="gsf")
@click.option("--acc_grad_iter", type=int, default=1)
@click.option("--enforce_train_epoch_size", type=int, default=None)
@click.option("--enforce_val_epoch_size", type=int, default=None)
@click.option("--gaussian_blur_kernel_size", type=int, default=5)
@click.option("--tdeed_arch_n_layers", type=int, default=2)
@click.option("--tdeed_arch_sgp_ks", type=int, default=5)
@click.option("--tdeed_arch_sgp_k", type=int, default=4)
@click.option("--model_checkpoint_path", type=str, default=None)
@click.option("--save_as", type=str, default="tdeed_competition.pt")
@click.option("--experiment_name", type=str, default="tdeed_competition")
@click.option("--val_dataset_path", type=str, default=None)
@click.option("--val_split", type=float, default=0.1)
@click.option("--random_seed", type=int, default=EXPERIMENTS_RANDOM_SEED)
@click.option("--use_wandb", type=bool, default=False)
@click.option("--num_workers", type=int, default=4)
@click.option("--freeze_backbone", type=bool, default=False)
@click.option("--weight_decay", type=float, default=0.0)
@click.option("--backbone_lr_scale", type=float, default=0.1)
@click.option("--class_weight_mode", type=click.Choice(["none", "inverse_sqrt"]), default="none")
@click.option("--class_weight_cap", type=float, default=3.0)
@click.option("--grad_checkpointing", type=bool, default=False)
@click.option(
    "--focal_loss_gamma",
    type=float,
    default=0.0,
    help="If > 0, use Focal Loss with this gamma (typical: 2.0). 0 = regular CE.",
)
@click.option(
    "--comp_score_snms_window",
    type=int,
    default=50,
    help="SNMS window (frames) used when eval_metric=competition_score.",
)
@click.option(
    "--comp_score_threshold",
    type=float,
    default=0.5,
    help="Confidence threshold used when eval_metric=competition_score.",
)
def train_competition(
    dataset_path: str,
    resolution: int = 224,
    clip_frames_count: int = 80,
    overlap: int = 68,
    displacement: int = 4,
    flip_proba: float = 0.2,
    camera_move_proba: float = 0.2,
    crop_proba: float = 0.2,
    even_choice_proba: float = 0.0,
    nr_epochs: int = 40,
    warm_up_epochs: int = 1,
    learning_rate: float = 0.0006,
    train_batch_size: int = 8,
    val_batch_size: int = 8,
    loss_foreground_weight: int = 5,
    eval_metric: str = "loss",
    start_eval_epoch_nr: int = 0,
    features_model_name: str = "regnety_008",
    temporal_shift_mode: str = "gsf",
    acc_grad_iter: int = 1,
    enforce_train_epoch_size: int = None,
    enforce_val_epoch_size: int = None,
    gaussian_blur_kernel_size: int = 5,
    tdeed_arch_n_layers: int = 2,
    tdeed_arch_sgp_ks: int = 5,
    tdeed_arch_sgp_k: int = 4,
    model_checkpoint_path: str = None,
    save_as: str = "tdeed_competition.pt",
    experiment_name: str = "tdeed_competition",
    val_dataset_path: str = None,
    val_split: float = 0.1,
    random_seed: int = EXPERIMENTS_RANDOM_SEED,
    use_wandb: bool = False,
    num_workers: int = 4,
    freeze_backbone: bool = False,
    weight_decay: float = 0.0,
    backbone_lr_scale: float = 0.1,
    class_weight_mode: str = "none",
    class_weight_cap: float = 3.0,
    grad_checkpointing: bool = False,
    focal_loss_gamma: float = 0.0,
    comp_score_snms_window: int = 50,
    comp_score_threshold: float = 0.5,
):
    assert resolution in [224, 720]
    assert eval_metric in ["loss", "map", "competition_score"], (
        "eval_metric must be loss, map, or competition_score"
    )
    if eval_metric == "competition_score":
        assert val_dataset_path is not None, (
            "eval_metric=competition_score requires --val_dataset_path (Labels-ball.json GT)."
        )
    if use_wandb:
        wandb.init(project=experiment_name, sync_tensorboard=True)

    videos = load_competition_videos(dataset_path, resolution=resolution, labels_enum=Action)
    all_clips = []
    for v in tqdm(videos, desc="Loading clips ..."):
        clips = v.get_clips(accepted_gap=2)
        for clip in clips:
            all_clips += clip.split(clip_frames_count=clip_frames_count, overlap=overlap)

    n_with_events = sum(1 for c in all_clips if c.has_events)
    print(f"Total sub-clips: {len(all_clips)}, with events: {n_with_events} ({100*n_with_events/len(all_clips):.1f}%)")

    all_dataset = TeamTDeedDataset(
        all_clips,
        labels_enum=Action,
        displacement=displacement,
        return_dict=True,
        flip_proba=flip_proba,
        camera_move_proba=camera_move_proba,
        crop_proba=crop_proba,
        even_choice_proba=even_choice_proba,
    )

    if val_dataset_path is not None:
        val_videos = load_competition_videos(val_dataset_path, resolution=resolution, labels_enum=Action)
        val_clips = []
        for v in tqdm(val_videos, desc="Loading val clips ..."):
            for clip in v.get_clips(accepted_gap=2):
                val_clips += clip.split(clip_frames_count=clip_frames_count, overlap=overlap)
        print(f"Val set (separate path): {len(val_videos)} videos, {len(val_clips)} sub-clips")
        val_dataset = TeamTDeedDataset(
            val_clips,
            labels_enum=Action,
            displacement=displacement,
            return_dict=True,
            flip_proba=0.0,
            camera_move_proba=0.0,
            crop_proba=0.0,
            even_choice_proba=0.0,
        )
        train_dataset = all_dataset
    else:
        n_matches = len(all_dataset.get_unique_matches())
        assert n_matches >= 2, f"Need at least 2 unique clips, got {n_matches}"
        n_val = max(1, round(n_matches * val_split))
        n_train = n_matches - n_val
        print(f"Splitting {n_matches} clips → train={n_train}, val={n_val}")
        train_dataset, val_dataset = all_dataset.split_by_matches(
            counts=[n_train, n_val], random_seed=random_seed
        )

    if enforce_train_epoch_size is not None:
        train_dataset.enforced_epoch_size = enforce_train_epoch_size
    if enforce_val_epoch_size is not None:
        val_dataset.enforced_epoch_size = enforce_val_epoch_size

    if eval_metric in ("map", "competition_score"):
        val_dataset.return_dict = False
    val_dataset.flip_proba = 0.0
    val_dataset.camera_move_proba = 0.0
    val_dataset.crop_proba = 0.0
    val_dataset.even_choice_proba = 0.0

    if class_weight_mode == "inverse_sqrt":
        per_class_weights, class_counts = compute_inverse_sqrt_class_weights(
            videos, labels_enum=Action, cap=class_weight_cap
        )
        print(f"Class weights (inverse_sqrt, cap={class_weight_cap}):")
        for cls, w, n in zip(Action, per_class_weights, class_counts):
            print(f"  {cls.name:<20s} n={n:<5d} weight={w:.3f}")
    else:
        per_class_weights = None
        print("Class weights: uniform foreground_weight for all event classes")

    tdeed_model = TDeedModule(
        clip_len=clip_frames_count,
        n_layers=tdeed_arch_n_layers,
        sgp_ks=tdeed_arch_sgp_ks,
        sgp_k=tdeed_arch_sgp_k,
        num_classes=len(Action),
        features_model_name=features_model_name,
        temporal_shift_mode=temporal_shift_mode,
        gaussian_blur_ks=gaussian_blur_kernel_size,
        grad_checkpointing=grad_checkpointing,
    )
    if model_checkpoint_path:
        tdeed_model.load_backbone(model_weight_path=model_checkpoint_path)
    if freeze_backbone:
        tdeed_model.freeze_backbone()
        trainable = sum(p.numel() for p in tdeed_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in tdeed_model.parameters())
        print(f"Backbone frozen. Trainable params: {trainable:,} / {total:,}")

    train_tdeed(
        experiment_name=experiment_name,
        model=tdeed_model,
        labels_enum=Action,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        eval_metric=eval_metric,
        nr_epochs=nr_epochs,
        start_eval_epoch_nr=start_eval_epoch_nr,
        foreground_weight=loss_foreground_weight,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        device=DEFAULT_DEVICE,
        acc_grad_iter=acc_grad_iter,
        save_as=save_as,
        lr=learning_rate,
        loss_weights=[1.5, 1],
        warm_up_epochs=warm_up_epochs,
        per_class_weights=per_class_weights,
        num_workers=num_workers,
        weight_decay=weight_decay,
        backbone_lr_scale=0.0 if freeze_backbone else backbone_lr_scale,
        focal_loss_gamma=focal_loss_gamma,
        comp_score_snms_window=comp_score_snms_window,
        comp_score_threshold=comp_score_threshold,
    )


def _build_video_result_json(
    vid_data,
    labels_enum: Type[enum.Enum],
    min_confidence: float = 0.0,
) -> dict:
    """Per-video JSON containing only predicted annotations, matching the
    Labels-ball.json ground-truth format.
    """
    video = vid_data.video
    scores = vid_data.scores
    class_list = list(labels_enum)
    fps = float(video.metadata_fps)

    annotations = []
    for class_idx, cls in enumerate(class_list):
        for f in np.where(scores[:, class_idx] > min_confidence)[0]:
            conf = float(scores[f, class_idx])
            position_ms = int(f / fps * 1000)
            seconds = position_ms / 1000
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            annotations.append(
                {
                    "gameTime": f"1 - {minutes}:{secs:02d}",
                    "label": cls.value,
                    "position": str(position_ms),
                    "team": "not applicable",
                    "visibility": "visible",
                    "confidence": conf,
                }
            )

    annotations.sort(key=lambda a: int(a["position"]))

    return {"annotations": annotations}


def _compute_per_class_ap(
    scored_videos,
    labels_enum: Type[enum.Enum],
    class_configs: dict,
    apply_min_score: bool = False,
) -> dict:
    """Per-class Average Precision using per-class tolerance_seconds.

    Uses each video's own fps (video.metadata_fps) to convert tolerance_seconds
    to frame tolerance.

    Returns dict {class_name: {"ap", "gt_count", "n_pred", "tolerance_s", "weight"}}.
    Background is not evaluated.
    """
    class_list = list(labels_enum)
    results: dict = {}

    video_fps_map = {v.video.absolute_path: float(v.video.metadata_fps) for v in scored_videos}

    for class_idx, cls in enumerate(class_list):
        cfg = class_configs.get(cls)
        tolerance_seconds = cfg.tolerance_seconds if cfg else 1.0
        weight = cfg.weight if cfg else 1.0
        min_score = cfg.min_score if (cfg and apply_min_score) else 0.0

        all_predictions = []
        all_ground_truths: dict = {}

        for vid_data in scored_videos:
            vid_id = vid_data.video.absolute_path
            class_preds = vid_data.scores[:, class_idx]
            class_targets = vid_data.targets[:, class_idx]

            pred_indices = np.where(class_preds > min_score)[0]
            pred_scores = class_preds[pred_indices]
            for idx, score in zip(pred_indices, pred_scores):
                all_predictions.append(
                    {"video_id": vid_id, "frame_idx": int(idx), "score": float(score)}
                )

            gt_indices = np.where(class_targets == 1)[0]
            if vid_id not in all_ground_truths:
                all_ground_truths[vid_id] = {
                    "gt_indices": gt_indices.tolist(),
                    "matches": np.zeros(len(gt_indices), dtype=bool),
                }
            else:
                all_ground_truths[vid_id]["gt_indices"].extend(gt_indices.tolist())
                all_ground_truths[vid_id]["matches"] = np.concatenate(
                    [
                        all_ground_truths[vid_id]["matches"],
                        np.zeros(len(gt_indices), dtype=bool),
                    ]
                )

        all_predictions.sort(key=lambda x: x["score"], reverse=True)
        TP = np.zeros(len(all_predictions))
        FP = np.zeros(len(all_predictions))
        total_gt = sum(len(v["gt_indices"]) for v in all_ground_truths.values())

        for idx, pred in enumerate(all_predictions):
            vid_id = pred["video_id"]
            frame_idx = pred["frame_idx"]
            delta_frames = int(round(tolerance_seconds * video_fps_map.get(vid_id, 25.0)))
            gt_info = all_ground_truths.get(vid_id, {"gt_indices": [], "matches": []})
            gt_indices = gt_info["gt_indices"]
            matches = gt_info["matches"]

            min_delta = float("inf")
            matched_gt_idx = -1
            for gt_idx, gt_frame in enumerate(gt_indices):
                if not matches[gt_idx]:
                    delta = abs(frame_idx - gt_frame)
                    if delta <= delta_frames and delta < min_delta:
                        min_delta = delta
                        matched_gt_idx = gt_idx

            if matched_gt_idx >= 0:
                TP[idx] = 1
                matches[matched_gt_idx] = True
                all_ground_truths[vid_id]["matches"] = matches
            else:
                FP[idx] = 1

        if total_gt > 0 and len(all_predictions) > 0:
            cum_TP = np.cumsum(TP)
            cum_FP = np.cumsum(FP)
            precisions = cum_TP / (cum_TP + cum_FP + 1e-8)
            recalls = cum_TP / (total_gt + 1e-8)
            ap = float(TDeedMAPEvaluator.compute_ap(recalls, precisions))
        else:
            ap = 0.0

        results[cls.name] = {
            "ap": ap,
            "gt_count": int(total_gt),
            "n_pred": len(all_predictions),
            "tolerance_s": tolerance_seconds,
            "weight": weight,
        }

    return results


def _gt_labels_path_for_video(video) -> str:
    """Labels-ball.json path for a SoccerVideo (same dir as the mp4)."""
    return os.path.join(os.path.dirname(video.absolute_path), "Labels-ball.json")


def _apply_single_class_snms(
    raw_scores: np.ndarray,
    class_idx: int,
    window: int,
    snms_threshold: float = 0.01,
) -> np.ndarray:
    """SNMS on a single class column. Replicates soft_non_maximum_suppression exactly
    (greedy peak pick with quadratic-distance suppression) but without tqdm, so the
    (class x window x video) sweep doesn't flood the logs with progress bars."""
    s = raw_scores[:, class_idx].astype(np.float32, copy=True)
    num_frames = s.shape[0]
    window = int(window)
    frames = np.arange(num_frames)
    processed = np.zeros(num_frames, dtype=bool)
    output_s = np.zeros(num_frames, dtype=np.float32)

    while True:
        s_masked = s.copy()
        s_masked[processed] = -np.inf
        e1_idx = int(np.argmax(s_masked))
        e1_score = float(s_masked[e1_idx])
        if e1_score < snms_threshold or e1_score == -np.inf:
            break
        output_s[e1_idx] = e1_score
        processed[e1_idx] = True

        distances = np.abs(frames - e1_idx)
        within = (distances <= window) & (~processed)
        if np.any(within):
            suppression = (distances[within] ** 2) / (window ** 2)
            s[within] = s[within] * suppression

    return output_s


def _class_events_from_post_snms(
    post_snms_1d: np.ndarray,
    class_label: str,
    fps: float,
    threshold: float,
) -> List[Event]:
    """Convert a post-SNMS 1-D score array into event predictions above threshold."""
    frames = np.where(post_snms_1d > threshold)[0]
    events: List[Event] = []
    for f in frames:
        conf = float(post_snms_1d[int(f)])
        time_ms = int(int(f) / fps * 1000)
        events.append(
            Event(label=class_label, time_ms=time_ms, frame_nr=int(f), confidence=conf)
        )
    return events


def _optimize_per_class_thresholds(
    scored_videos,
    video_gt_events: List[List[Event]],
    labels_enum: Type[enum.Enum],
    threshold_sweep: List[float],
    snms_window_sweep: List[int],
    snms_base_threshold: float = 0.01,
) -> dict:
    """For each trained class, sweep (snms_window, threshold) and find the combo that
    maximises total class contribution (matched_points - fp_penalty) across videos.

    Per-class independence holds under the competition scoring formula: a class's
    matched_points / fp_penalty only depend on that class's predictions and GT.
    The final-score clamp (max(0, raw)) only interacts when a video's other classes
    are strongly negative; in practice most videos have positive raw score.

    Returns a dict keyed by class name with the optimal setting and diagnostics.
    """
    class_list = list(labels_enum)
    video_fps = [float(vd.video.metadata_fps) for vd in scored_videos]

    results: dict = {}

    for class_idx, cls in enumerate(tqdm(class_list, desc="Sweeping classes")):
        class_label = cls.value  # e.g. "pass"
        if class_label not in COMPETITION_ACTIONS:
            # Shouldn't happen for Action enum but be safe.
            continue

        gt_count_total = sum(
            sum(1 for e in gt if e.label == class_label) for gt in video_gt_events
        )

        # Track the grid of (window, threshold) -> contribution
        best = {
            "best_window": None,
            "best_threshold": None,
            "best_contribution": -float("inf"),
            "matched_points": 0.0,
            "fp_penalty": 0.0,
            "n_matched": 0,
            "n_fp": 0,
            "n_gt": gt_count_total,
            "tp_confidences": [],
            "fp_confidences": [],
            "grid": {},
        }

        for window in snms_window_sweep:
            # Pre-compute post-SNMS for this class across all videos once per window.
            post_snms_per_video: List[np.ndarray] = []
            for vd in scored_videos:
                post = _apply_single_class_snms(
                    vd.scores, class_idx, window, snms_base_threshold
                )
                post_snms_per_video.append(post)

            for thr in threshold_sweep:
                total_matched = 0.0
                total_fp_pen = 0.0
                total_n_matched = 0
                total_n_fp = 0
                tp_confs: List[float] = []
                fp_confs: List[float] = []

                for v_idx, post in enumerate(post_snms_per_video):
                    pred_events = _class_events_from_post_snms(
                        post,
                        class_label=class_label,
                        fps=video_fps[v_idx],
                        threshold=thr,
                    )
                    m_pts, fp_pen, n_m, n_fp, _ = score_class_on_video(
                        video_gt_events[v_idx],
                        pred_events,
                        class_label=class_label,
                    )
                    total_matched += m_pts
                    total_fp_pen += fp_pen
                    total_n_matched += n_m
                    total_n_fp += n_fp

                    # Re-walk predictions to tag TP vs FP for confidence histograms.
                    # (Re-match on this subset; same logic as score_class_on_video.)
                    weight, tol_s = COMPETITION_ACTIONS[class_label]
                    tol_ms = tol_s * 1000.0
                    gt_of_class = [
                        e for e in video_gt_events[v_idx] if e.label == class_label
                    ]
                    matched_pred_idx: set = set()
                    for gt in gt_of_class:
                        best_j, best_diff = None, float("inf")
                        for j, pr in enumerate(pred_events):
                            if j in matched_pred_idx:
                                continue
                            diff = abs(pr.time_ms - gt.time_ms)
                            if diff <= tol_ms and diff < best_diff:
                                best_j, best_diff = j, diff
                        if best_j is not None:
                            matched_pred_idx.add(best_j)
                            tp_confs.append(pred_events[best_j].confidence or 0.0)
                    for j, pr in enumerate(pred_events):
                        if j not in matched_pred_idx:
                            fp_confs.append(pr.confidence or 0.0)

                contribution = total_matched - total_fp_pen
                best["grid"][(int(window), float(thr))] = {
                    "contribution": float(contribution),
                    "matched_points": float(total_matched),
                    "fp_penalty": float(total_fp_pen),
                    "n_matched": int(total_n_matched),
                    "n_fp": int(total_n_fp),
                }
                if contribution > best["best_contribution"]:
                    best["best_contribution"] = float(contribution)
                    best["best_window"] = int(window)
                    best["best_threshold"] = float(thr)
                    best["matched_points"] = float(total_matched)
                    best["fp_penalty"] = float(total_fp_pen)
                    best["n_matched"] = int(total_n_matched)
                    best["n_fp"] = int(total_n_fp)
                    best["tp_confidences"] = list(tp_confs)
                    best["fp_confidences"] = list(fp_confs)

        results[cls.name] = best

    return results


def _conf_percentiles(confidences: List[float]) -> dict:
    if not confidences:
        return {"n": 0, "p10": None, "p50": None, "p90": None, "min": None, "max": None}
    arr = np.asarray(confidences, dtype=np.float32)
    return {
        "n": int(arr.size),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _score_videos_with_optimal(
    scored_videos,
    video_gt_events: List[List[Event]],
    labels_enum: Type[enum.Enum],
    optimal: dict,
    snms_base_threshold: float = 0.01,
) -> List[dict]:
    """Apply each class's optimal (window, threshold) to build final predictions per
    video and compute the competition score. Classes with non-positive best
    contribution are dropped from submission."""
    class_list = list(labels_enum)
    per_video: List[dict] = []

    for v_idx, vd in enumerate(scored_videos):
        fps = float(vd.video.metadata_fps)
        pred_events: List[Event] = []
        for class_idx, cls in enumerate(class_list):
            info = optimal.get(cls.name)
            if info is None:
                continue
            if info["best_contribution"] <= 0.0:
                continue  # DO NOT SUBMIT
            post = _apply_single_class_snms(
                vd.scores, class_idx, info["best_window"], snms_base_threshold
            )
            pred_events.extend(
                _class_events_from_post_snms(
                    post, class_label=cls.value, fps=fps, threshold=info["best_threshold"]
                )
            )
        result = score_video(video_gt_events[v_idx], pred_events)
        per_video.append(
            {
                "video_path": vd.video.absolute_path,
                "result": result,
                "n_pred": len(pred_events),
            }
        )
    return per_video


@cli.command()
@click.option("--val_dataset_path", type=str, required=True)
@click.option("--model_checkpoint_path", type=str, required=True)
@click.option("--resolution", type=int, default=224)
@click.option("--clip_frames_count", type=int, default=170)
@click.option("--overlap", type=int, default=136)
@click.option("--displacement", type=int, default=4)
@click.option("--val_batch_size", type=int, default=2)
@click.option("--features_model_name", type=str, default="regnety_008")
@click.option("--temporal_shift_mode", type=str, default="gsf")
@click.option("--tdeed_arch_n_layers", type=int, default=2)
@click.option("--tdeed_arch_sgp_ks", type=int, default=5)
@click.option("--tdeed_arch_sgp_k", type=int, default=4)
@click.option("--gaussian_blur_kernel_size", type=int, default=5)
@click.option("--use_snms", type=bool, default=True)
@click.option("--snms_window", type=int, default=12, help="Uniform SNMS window (frames). Ignored if --snms_windows is set.")
@click.option("--snms_threshold", type=float, default=0.01)
@click.option(
    "--snms_windows",
    type=str,
    default=None,
    help=(
        "Per-class SNMS windows, comma-separated in Action enum order "
        "(12 ints). Example: '50,50,50,50,50,75,75,50,50,50,75,75'. "
        "Overrides --snms_window."
    ),
)
@click.option("--apply_min_score", type=bool, default=False)
@click.option("--output_dir", type=str, default=None, help="If set, write per-video predictions JSON here (Labels-ball.json format).")
@click.option("--min_confidence", type=float, default=0.0, help="Drop predictions below this confidence from the output JSON.")
@click.option(
    "--optimize_thresholds",
    is_flag=True,
    default=False,
    help=(
        "Sweep per-class (confidence threshold, SNMS window) to maximise total "
        "competition score (score.py formula) on the validation set. Overrides "
        "the mAP evaluation path and runs inference WITHOUT pre-applied SNMS."
    ),
)
@click.option(
    "--threshold_sweep",
    type=str,
    default="0.05,0.95,0.05",
    help="Threshold sweep as start,stop,step (inclusive of start; stop is exclusive).",
)
@click.option(
    "--snms_window_sweep",
    type=str,
    default="25,50,75,100",
    help="Comma-separated list of SNMS window sizes to sweep over.",
)
def evaluate_competition(
    val_dataset_path: str,
    model_checkpoint_path: str,
    resolution: int = 224,
    clip_frames_count: int = 170,
    overlap: int = 136,
    displacement: int = 4,
    val_batch_size: int = 2,
    features_model_name: str = "regnety_008",
    temporal_shift_mode: str = "gsf",
    tdeed_arch_n_layers: int = 2,
    tdeed_arch_sgp_ks: int = 5,
    tdeed_arch_sgp_k: int = 4,
    gaussian_blur_kernel_size: int = 5,
    use_snms: bool = True,
    snms_window: int = 12,
    snms_threshold: float = 0.01,
    snms_windows: str = None,
    apply_min_score: bool = False,
    output_dir: str = None,
    min_confidence: float = 0.0,
    optimize_thresholds: bool = False,
    threshold_sweep: str = "0.05,0.95,0.05",
    snms_window_sweep: str = "25,50,75,100",
):
    """Evaluate a competition model on a validation set with per-class AP.

    Loads videos from --val_dataset_path with Action labels, runs inference,
    applies SNMS, and computes per-class AP at each class's tolerance_seconds
    (from ACTION_CONFIGS). Reports three aggregates:
      - mAP unweighted (simple mean across classes)
      - mAP weighted by support (sample-count weighted)
      - competition-weighted mAP (using ACTION_CONFIGS weight per class)
    """
    assert resolution in [224, 720]

    print(f"Loading competition videos from {val_dataset_path} ...")
    videos = load_competition_videos(
        val_dataset_path, resolution=resolution, labels_enum=Action
    )
    print(f"Loaded {len(videos)} videos")

    all_clips = []
    for v in tqdm(videos, desc="Building clips"):
        clips = v.get_clips(accepted_gap=2)
        for clip in clips:
            all_clips += clip.split(
                clip_frames_count=clip_frames_count, overlap=overlap
            )
    print(f"Total clips: {len(all_clips)}")

    val_dataset = TeamTDeedDataset(
        all_clips,
        labels_enum=Action,
        displacement=displacement,
        return_dict=False,
        flip_proba=0.0,
        camera_move_proba=0.0,
        crop_proba=0.0,
        even_choice_proba=0.0,
    )

    print(f"Building model (num_classes={len(Action)}) ...")
    tdeed_model = TDeedModule(
        clip_len=clip_frames_count,
        n_layers=tdeed_arch_n_layers,
        sgp_ks=tdeed_arch_sgp_ks,
        sgp_k=tdeed_arch_sgp_k,
        num_classes=len(Action),
        features_model_name=features_model_name,
        temporal_shift_mode=temporal_shift_mode,
        gaussian_blur_ks=gaussian_blur_kernel_size,
    )
    print(f"Loading weights from {model_checkpoint_path} ...")
    tdeed_model.load_all(model_weight_path=model_checkpoint_path)
    tdeed_model = tdeed_model.to(DEFAULT_DEVICE)
    tdeed_model.eval()

    evaluator = BASTeamTDeedEvaluator(
        model=tdeed_model,
        dataset=val_dataset,
        delta_frames_tolerance=1,
    )

    if optimize_thresholds:
        # Inference WITHOUT SNMS: we sweep SNMS window per-class in post-processing.
        print("Running inference (raw scores, no SNMS) for threshold optimization ...")
        scored_videos = evaluator.get_scored_videos(
            batch_size=val_batch_size,
            use_snms=False,
            use_hflip=False,
            snms_params=None,
        )

        # Parse sweep configuration (inclusive of start AND stop when reachable).
        try:
            t_start, t_stop, t_step = [float(x) for x in threshold_sweep.split(",")]
        except Exception as e:
            raise click.BadParameter(
                f"--threshold_sweep must be 'start,stop,step' floats; got '{threshold_sweep}'"
            ) from e
        thresholds: List[float] = []
        t = t_start
        while t <= t_stop + 1e-9:
            thresholds.append(round(t, 6))
            t += t_step
        if not thresholds:
            thresholds = [t_start]
        windows: List[int] = [
            int(x.strip()) for x in snms_window_sweep.split(",") if x.strip()
        ]
        if not windows:
            raise click.BadParameter("--snms_window_sweep produced empty list")

        print(
            f"Threshold sweep: {len(thresholds)} values in [{thresholds[0]:.2f}, {thresholds[-1]:.2f}]; "
            f"SNMS windows: {windows}"
        )

        # Load ALL-15-class GT per video for correct denominators.
        print("Loading full 15-class GT per video ...")
        video_gt_events: List[List[Event]] = []
        for vd in scored_videos:
            gt_path = _gt_labels_path_for_video(vd.video)
            video_gt_events.append(load_all_gt_events(gt_path))

        print("Sweeping per-class (SNMS window, threshold) ...")
        optimal = _optimize_per_class_thresholds(
            scored_videos=scored_videos,
            video_gt_events=video_gt_events,
            labels_enum=Action,
            threshold_sweep=thresholds,
            snms_window_sweep=windows,
            snms_base_threshold=snms_threshold,
        )

        # Report per-class results.
        print("")
        print("=" * 110)
        print(
            f"{'Class':<18} {'w*':>4} {'thr*':>6} {'contrib':>10} "
            f"{'matched':>9} {'FP_pen':>9} {'nM':>4} {'nFP':>5} {'nGT':>4} "
            f"{'TPp50':>7} {'FPp50':>7} {'submit?':>8}"
        )
        print("-" * 110)
        total_contribution = 0.0
        for cls in Action:
            info = optimal.get(cls.name)
            if info is None:
                continue
            tp_stats = _conf_percentiles(info["tp_confidences"])
            fp_stats = _conf_percentiles(info["fp_confidences"])
            submit = info["best_contribution"] > 0.0
            if submit:
                total_contribution += info["best_contribution"]
            tp_p50 = f"{tp_stats['p50']:.3f}" if tp_stats["p50"] is not None else "   -"
            fp_p50 = f"{fp_stats['p50']:.3f}" if fp_stats["p50"] is not None else "   -"
            print(
                f"{cls.name:<18} "
                f"{info['best_window']:>4} "
                f"{info['best_threshold']:>6.2f} "
                f"{info['best_contribution']:>10.2f} "
                f"{info['matched_points']:>9.2f} "
                f"{info['fp_penalty']:>9.2f} "
                f"{info['n_matched']:>4d} "
                f"{info['n_fp']:>5d} "
                f"{info['n_gt']:>4d} "
                f"{tp_p50:>7} "
                f"{fp_p50:>7} "
                f"{'YES' if submit else 'DROP':>8}"
            )
        print("-" * 110)

        print("Scoring videos with optimal per-class (window, threshold) ...")
        per_video = _score_videos_with_optimal(
            scored_videos=scored_videos,
            video_gt_events=video_gt_events,
            labels_enum=Action,
            optimal=optimal,
            snms_base_threshold=snms_threshold,
        )
        agg = aggregate_final_scores([pv["result"] for pv in per_video])

        print("")
        print("=" * 110)
        print(f"{'Video':<60} {'raw':>8} {'final':>8} {'n_pred':>7} {'n_gt':>5}")
        print("-" * 110)
        for pv in per_video:
            r = pv["result"]
            vid_name = os.path.basename(os.path.dirname(pv["video_path"]))
            n_gt = sum(1 for _ in r.rows if _.kind in ("MATCH", "MISS"))
            print(
                f"{vid_name:<60} "
                f"{r.raw_score:>8.4f} "
                f"{r.final_score:>8.4f} "
                f"{pv['n_pred']:>7d} "
                f"{n_gt:>5d}"
            )
        print("-" * 110)
        print(
            f"Videos: {agg['n_videos']}  |  "
            f"avg raw: {agg['avg_raw_score']:.4f}  |  "
            f"avg final (clamped): {agg['avg_final_score']:.4f}  |  "
            f"sum matched: {agg['sum_matched']:.2f}  |  "
            f"sum FP penalty: {agg['sum_fp_penalty']:.2f}  |  "
            f"sum GT weight: {agg['sum_gt_weight']:.2f}"
        )
        print("=" * 110)
        print(
            f"\n>>> COMPETITION SCORE (avg of per-video clamped finals): "
            f"{agg['avg_final_score']*100:.2f}%  <<<"
        )
        print(f"    (current leader is at ~45%)")

        # Optional: write per-video result JSONs to output_dir, plus a summary with
        # the optimal parameters so we can re-use them.
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            print(f"\nWriting optimized results to: {output_dir}")
            opt_summary = {
                "model_checkpoint": model_checkpoint_path,
                "val_dataset_path": val_dataset_path,
                "n_videos": agg["n_videos"],
                "avg_final_score": agg["avg_final_score"],
                "avg_raw_score": agg["avg_raw_score"],
                "sum_matched": agg["sum_matched"],
                "sum_fp_penalty": agg["sum_fp_penalty"],
                "sum_gt_weight": agg["sum_gt_weight"],
                "per_class": {
                    cls.name: {
                        "best_window": optimal[cls.name]["best_window"],
                        "best_threshold": optimal[cls.name]["best_threshold"],
                        "best_contribution": optimal[cls.name]["best_contribution"],
                        "matched_points": optimal[cls.name]["matched_points"],
                        "fp_penalty": optimal[cls.name]["fp_penalty"],
                        "n_matched": optimal[cls.name]["n_matched"],
                        "n_fp": optimal[cls.name]["n_fp"],
                        "n_gt": optimal[cls.name]["n_gt"],
                        "submit": optimal[cls.name]["best_contribution"] > 0.0,
                        "tp_confidence_stats": _conf_percentiles(
                            optimal[cls.name]["tp_confidences"]
                        ),
                        "fp_confidence_stats": _conf_percentiles(
                            optimal[cls.name]["fp_confidences"]
                        ),
                    }
                    for cls in Action
                    if cls.name in optimal
                },
                "per_video": [
                    {
                        "video_path": pv["video_path"],
                        "raw_score": pv["result"].raw_score,
                        "final_score": pv["result"].final_score,
                        "matched_score": pv["result"].matched_score,
                        "fp_penalty": pv["result"].fp_penalty,
                        "total_gt_weight": pv["result"].total_gt_weight,
                        "n_pred": pv["n_pred"],
                    }
                    for pv in per_video
                ],
            }
            with open(os.path.join(output_dir, "_optimization_summary.json"), "w") as f:
                json.dump(opt_summary, f, indent=2)
            print(f"Wrote _optimization_summary.json")

        return optimal

    if snms_windows is not None:
        per_class = [int(x.strip()) for x in snms_windows.split(",") if x.strip()]
        num_action_classes = len(Action)
        assert len(per_class) == num_action_classes, (
            f"--snms_windows has {len(per_class)} values, expected {num_action_classes} "
            f"(one per Action class). Order: {[a.name for a in Action]}"
        )
        snms_class_window = per_class
        print(
            "Using per-class SNMS windows: "
            + ", ".join(f"{a.name}={w}" for a, w in zip(Action, per_class))
        )
    else:
        snms_class_window = snms_window
        print(f"Using uniform SNMS window: {snms_window}")

    print("Running inference ...")
    scored_videos = evaluator.get_scored_videos(
        batch_size=val_batch_size,
        use_snms=use_snms,
        use_hflip=False,
        snms_params=dict(class_window=snms_class_window, threshold=snms_threshold),
    )

    print("Computing per-class AP ...")
    per_class = _compute_per_class_ap(
        scored_videos=scored_videos,
        labels_enum=Action,
        class_configs=ACTION_CONFIGS,
        apply_min_score=apply_min_score,
    )

    aps = [v["ap"] for v in per_class.values()]
    supports = [v["gt_count"] for v in per_class.values()]
    weights = [v["weight"] for v in per_class.values()]

    map_unweighted = float(np.mean(aps)) if aps else 0.0
    total_support = sum(supports)
    map_support_weighted = (
        float(sum(a * s for a, s in zip(aps, supports)) / total_support)
        if total_support > 0
        else 0.0
    )
    total_weight = sum(weights)
    map_competition_weighted = (
        float(sum(a * w for a, w in zip(aps, weights)) / total_weight)
        if total_weight > 0
        else 0.0
    )

    print("")
    print("=" * 88)
    print(f"{'Class':<18} {'AP':>7} {'GT':>6} {'Preds':>8} {'tol(s)':>7} {'weight':>7}")
    print("-" * 88)
    for cls_name, info in per_class.items():
        print(
            f"{cls_name:<18} "
            f"{info['ap']*100:>6.2f}% "
            f"{info['gt_count']:>6d} "
            f"{info['n_pred']:>8d} "
            f"{info['tolerance_s']:>6.1f}s "
            f"{info['weight']:>7.2f}"
        )
    print("-" * 88)
    print(f"{'mAP (unweighted)':<35} {map_unweighted*100:>6.2f}%")
    print(f"{'mAP (support-weighted)':<35} {map_support_weighted*100:>6.2f}%")
    print(f"{'mAP (competition-weighted)':<35} {map_competition_weighted*100:>6.2f}%")
    print("=" * 88)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nWriting per-video JSON results to: {output_dir}")

        summary = {
            "model_checkpoint": model_checkpoint_path,
            "val_dataset_path": val_dataset_path,
            "num_videos": len(scored_videos),
            "mAP_unweighted": map_unweighted,
            "mAP_support_weighted": map_support_weighted,
            "mAP_competition_weighted": map_competition_weighted,
            "per_class": {
                cls_name: {
                    "ap": info["ap"],
                    "gt_count": info["gt_count"],
                    "n_pred": info["n_pred"],
                    "tolerance_s": info["tolerance_s"],
                    "weight": info["weight"],
                }
                for cls_name, info in per_class.items()
            },
        }
        with open(os.path.join(output_dir, "_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        for vid_data in scored_videos:
            try:
                video_json = _build_video_result_json(
                    vid_data=vid_data,
                    labels_enum=Action,
                    min_confidence=min_confidence,
                )
            except Exception as e:
                print(f"  [warn] failed to build JSON for {vid_data.video.absolute_path}: {e}")
                continue

            stem = os.path.splitext(os.path.basename(vid_data.video.absolute_path))[0]
            parent = os.path.basename(os.path.dirname(vid_data.video.absolute_path))
            out_name = f"{parent}__{stem}.json" if parent else f"{stem}.json"
            out_path = os.path.join(output_dir, out_name)
            with open(out_path, "w") as f:
                json.dump(video_json, f, indent=2)

        print(f"Wrote {len(scored_videos)} per-video JSON files + _summary.json")

    return per_class


if __name__ == "__main__":
    # Single GPU:  uv run python dudek/scripts/tdeed.py pretrain ...
    # Multi-GPU:   uv run torchrun --nproc_per_node=N dudek/scripts/tdeed.py pretrain ...
    import traceback

    _rank = int(os.environ.get("RANK", 0))
    _log_dir = os.environ.get("TRAIN_LOG_DIR", "logs")
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, f"rank_{_rank}.log")

    class _Tee:
        def __init__(self, stream, path):
            self._stream = stream
            self._file = open(path, "w", buffering=1)

        def write(self, data):
            self._stream.write(data)
            self._file.write(data)

        def flush(self):
            self._stream.flush()
            self._file.flush()

        def __getattr__(self, attr):
            return getattr(self._stream, attr)

    import sys
    sys.stdout = _Tee(sys.stdout, _log_path)
    sys.stderr = _Tee(sys.stderr, _log_path)

    try:
        cli()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        raise