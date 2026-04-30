import dataclasses
import enum
import json
import os
import time
from datetime import timedelta
from typing import Type, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext

from torch.optim.lr_scheduler import (
    LRScheduler,
)
import torch.nn.functional as F

from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from dudek.config import DEFAULT_DEVICE
from dudek.data.team_bas import BASLabel, ActionLabel
from dudek.ml.data.tdeed import TeamTDeedDataset
from dudek.ml.model.tdeed.eval.two_heads import BASTeamTDeedEvaluator


from dudek.ml.model.tdeed.modules.tdeed import TDeedModule

from dudek.utils.ml import get_lr_scheduler_with_warmup
from dudek.utils.competition_score import (
    Event,
    aggregate_final_scores,
    load_all_gt_events,
    score_video,
)
from dudek.utils.common import soft_non_maximum_suppression


@dataclasses.dataclass
class TDeedLoss:
    total_loss: float
    ce_labels_loss: float
    mse_displacement_loss: float
    ce_per_class: Optional[dict] = None


def train(
    experiment_name: str,
    model: TDeedModule,
    labels_enum: Type[BASLabel | ActionLabel],
    train_dataset: TeamTDeedDataset,
    val_dataset: Optional[TeamTDeedDataset],
    eval_metric: str,
    nr_epochs: int,
    start_eval_epoch_nr: int,
    device=DEFAULT_DEVICE,
    foreground_weight: int = 5,
    train_batch_size: int = 4,
    val_batch_size: int = 8,
    acc_grad_iter: int = 1,
    warm_up_epochs: int = 1,
    save_as: Optional[str] = "best.pt",
    save_every_epoch: bool = False,
    lr: float = 0.0004,
    loss_weights=None,
    per_class_weights=None,
    num_workers: int = 4,
    weight_decay: float = 0.0,
    backbone_lr_scale: float = 1.0,
    focal_loss_gamma: float = 0.0,
    comp_score_snms_window: int = 50,
    comp_score_threshold: float = 0.5,
    use_bf16: bool = False,
    seq_freeze_epochs: int = 0,
    max_train_iter_per_epoch: Optional[int] = None,
):

    assert eval_metric in ["loss", "map", "competition_score"], (
        "eval_metric must be one of: loss, map, competition_score"
    )

    # DDP setup: torchrun sets LOCAL_RANK; plain python leaves it unset (-1).
    _local_rank = int(os.environ.get("LOCAL_RANK", -1))
    _is_ddp = _local_rank >= 0

    if _is_ddp:
        if not dist.is_initialized():
            # IMPORTANT: bump NCCL collective timeout to 2 hours.
            # Reason: competition_score evaluation runs on rank 0 only
            # (per-video sliding-window inference) and can take 30-60 minutes
            # for the full val set. Other ranks block at the next training
            # epoch's allreduce while rank 0 is busy. The default 10-minute
            # NCCL timeout fires and kills the run. 2 hours gives ample
            # margin for slow eval epochs without masking real hangs.
            dist.init_process_group(
                backend="nccl",
                timeout=timedelta(hours=2),
            )
        device = f"cuda:{_local_rank}"
        torch.cuda.set_device(_local_rank)

    _is_main = (not _is_ddp) or dist.get_rank() == 0

    model.to(device)

    # Phase 1 of the sequence-decoder warm-start: optionally freeze backbone
    # + SGP for the first ``seq_freeze_epochs`` epochs so the randomly-
    # initialised sequence decoder (and the per-frame heads) settle without
    # their gradients flowing back through and corrupting the BAS-pretrained
    # ``_features`` / ``_temp_fine`` weights. After phase 1 the optimizer is
    # rebuilt to include the unfrozen modules and the model is re-wrapped in
    # DDP so the reducer's bucket layout matches the new requires_grad set.
    #
    # IMPORTANT: freezing must happen BEFORE the DDP wrap. DDP captures the
    # set of params requiring gradients at construction time; toggling
    # requires_grad after the wrap leads to the reducer waiting for
    # gradients that never arrive (or vice versa).
    has_seq_decoder = getattr(model, "_seq_decoder", None) is not None
    if seq_freeze_epochs > 0:
        if not has_seq_decoder and _is_main:
            print(
                f"NOTE: --seq_freeze_epochs={seq_freeze_epochs} requested but "
                "the model has no sequence decoder; freezing backbone + SGP "
                "anyway, but you probably want use_seq_decoder=True."
            )
        model.freeze_backbone()
        model.freeze_sgp()
        if _is_main:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(
                f"[warm-start phase 1] backbone + SGP frozen for "
                f"{seq_freeze_epochs} epoch(s). Trainable: {trainable:,} / {total:,}"
            )

    if _is_ddp:
        model = DDP(model, device_ids=[_local_rank], output_device=_local_rank)
        if _is_main:
            print(f"[DDP] {dist.get_world_size()} GPUs active")
    elif torch.cuda.device_count() > 1:
        print(
            f"NOTE: {torch.cuda.device_count()} GPUs available but only 1 in use.\n"
            "Run with:  torchrun --nproc_per_node=2 dudek/scripts/tdeed.py"
        )

    raw_model = model.module if isinstance(model, DDP) else model

    # bf16 has the same dynamic range as fp32, so GradScaler is only required
    # for fp16 (where small gradients underflow to zero). On Blackwell-class
    # GPUs (RTX 5090 / H100) bf16 is also faster for attention / softmax,
    # which matters once the SequenceTransformerHead is enabled.
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = (
        None
        if use_bf16
        else (torch.amp.GradScaler("cuda") if "cuda" in device else None)
    )

    if _is_ddp:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    # NOTE: prefetch_factor=1 (default is 2) to keep host RAM bounded at 540p.
    # Each worker holds (prefetch_factor) batches in flight; with 24 workers
    # across 4 ranks at 170-frame 540p clips, prefetch_factor=2 alone costs
    # ~25 GB host RAM. 1 is the minimum legal value when num_workers > 0.
    _loader_extra = {"prefetch_factor": 1} if num_workers > 0 else {}
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        **_loader_extra,
    )
    if val_dataset is not None:
        eval_data_loader = DataLoader(
            val_dataset, batch_size=val_batch_size, shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            **_loader_extra,
        )

    optimizer_steps_per_epoch = max(len(train_data_loader) // acc_grad_iter, 1)

    def _build_optimizer_and_scheduler(epochs_remaining: int):
        """(Re)build optimizer + scheduler over currently-trainable params.

        Used at start, and again when leaving the freeze phase to bring
        backbone + SGP back into the optimizer. Optimizer momentum state is
        intentionally reset on rebuild — the post-freeze backbone has been
        idle for ``seq_freeze_epochs`` and we want a clean curve from there.
        """
        backbone_lr = lr * backbone_lr_scale
        param_groups = [
            {
                "params": [
                    p for p in raw_model._features.parameters() if p.requires_grad
                ],
                "lr": backbone_lr,
            },
            {
                "params": [
                    p
                    for name, p in raw_model.named_parameters()
                    if not name.startswith("_features") and p.requires_grad
                ],
                "lr": lr,
            },
        ]
        # Drop empty groups (e.g. backbone group is empty during phase 1).
        param_groups = [g for g in param_groups if len(g["params"]) > 0]
        opt = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        sch = get_lr_scheduler_with_warmup(
            opt,
            warm_up_steps=optimizer_steps_per_epoch * warm_up_epochs,
            total_training_steps=max(epochs_remaining - warm_up_epochs, 1)
            * optimizer_steps_per_epoch,
        )
        if _is_main:
            n_back = sum(g["params"][0].numel() for g in param_groups[:1]) if param_groups else 0
            print(
                f"Optimizer: backbone_lr={backbone_lr:.2e}  "
                f"head_lr={lr:.2e}  weight_decay={weight_decay}  "
                f"groups={len(param_groups)} (backbone params in opt: {n_back > 0})"
            )
        return opt, sch

    optimizer, scheduler = _build_optimizer_and_scheduler(nr_epochs)

    best_eval_metric = float("inf") if eval_metric == "loss" else 0.0
    evaluator = None
    if eval_metric == "map" and val_dataset is not None:
        evaluator = BASTeamTDeedEvaluator(
            model=model,
            dataset=val_dataset,
            delta_frames_tolerance=5,
        )
    elif eval_metric == "competition_score" and val_dataset is not None:
        # Need scores per frame per class for post-processed matching.
        val_dataset.return_dict = False
        evaluator = BASTeamTDeedEvaluator(
            model=model,
            dataset=val_dataset,
            delta_frames_tolerance=1,
        )

    if _is_main:
        hparam_dict = {
            "eval_metric": eval_metric,
            "foreground_weight": foreground_weight,
            "train_batch_size": train_batch_size,
            "val_batch_size": val_batch_size,
            "acc_grad_iter": acc_grad_iter,
            "start_eval_epoch_nr": start_eval_epoch_nr,
            "nr_epochs": nr_epochs,
            "displacement": train_dataset.displacement,
            "clip_length": len(train_dataset.clips[0].frames),
            "epoch_volume": len(train_dataset.clips),
            "horizontal_flip_proba": train_dataset.flip_proba,
            "crop_proba": train_dataset.crop_proba,
            "features_model_name": raw_model.features_model_name,
            "temporal_shift_mode": raw_model.temporal_shift_mode,
            "sgp_ks": raw_model.sgp_ks,
            "sgp_k": raw_model.sgp_k,
            "n_layers": raw_model.n_layers,
            "camera_move_proba": train_dataset.camera_move_proba,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "optimizer": optimizer.__class__.__name__,
            "model": raw_model.__class__.__name__,
            "train_dataset": train_dataset.__class__.__name__,
            "val_dataset": (
                val_dataset.__class__.__name__ if val_dataset is not None else None
            ),
            "device": str(device),
        }
        summary_writer = SummaryWriter(log_dir=f"runs/{experiment_name}_{time.time()}")
        summary_writer.add_text(
            "train/hyperparameters", json.dumps(hparam_dict, indent=4), global_step=0
        )
    else:
        summary_writer = None

    for epoch_nr in range(nr_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_nr)

        # End of phase 1 (freeze): unfreeze backbone + SGP, re-wrap DDP so
        # the reducer picks up the newly-trainable params, and rebuild the
        # optimizer / scheduler. We do this at the *start* of epoch
        # ``seq_freeze_epochs`` so that this is the first epoch where
        # backbone + SGP gradients flow.
        if seq_freeze_epochs > 0 and epoch_nr == seq_freeze_epochs:
            raw_model.unfreeze_backbone()
            raw_model.unfreeze_sgp()
            if _is_ddp:
                # Rebuild DDP because reducer buckets were built when backbone
                # + SGP had requires_grad=False; they are not in the reducer.
                model = DDP(
                    raw_model,
                    device_ids=[_local_rank],
                    output_device=_local_rank,
                )
                raw_model = model.module
            epochs_remaining = nr_epochs - epoch_nr
            optimizer, scheduler = _build_optimizer_and_scheduler(epochs_remaining)
            if _is_main:
                trainable = sum(
                    p.numel() for p in raw_model.parameters() if p.requires_grad
                )
                total = sum(p.numel() for p in raw_model.parameters())
                print(
                    f"[warm-start phase 2] epoch {epoch_nr}: backbone + SGP "
                    f"unfrozen. Trainable: {trainable:,} / {total:,}"
                )

        _go_through_epoch_train(
            model=model,
            labels_enum=labels_enum,
            data_loader=train_data_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            acc_grad_iter=acc_grad_iter,
            summary_writer=summary_writer,
            foreground_weight=foreground_weight,
            epoch_nr=epoch_nr,
            loss_weights=loss_weights or [1.5, 1],
            is_main=_is_main,
            per_class_weights=per_class_weights,
            focal_loss_gamma=focal_loss_gamma,
            autocast_dtype=autocast_dtype,
            max_iter=max_train_iter_per_epoch,
        )

        # Eval orchestration:
        #  - competition_score runs on ALL ranks (videos are sharded across
        #    ranks; results are gathered inside _eval_competition_score). This
        #    is critical: with single-rank eval, ranks 1+ block on the next
        #    train allreduce and NCCL times out.
        #  - loss and map eval still run rank-0-only (legacy paths). All ranks
        #    must still flip model.eval()/model.train() symmetrically and the
        #    non-main ranks idle through the eval block; for these paths this
        #    relies on the bumped 2-hour NCCL timeout we set at init.
        if val_dataset is not None and epoch_nr >= start_eval_epoch_nr:
            model.eval()
            if eval_metric == "competition_score":
                comp = _eval_competition_score(
                    evaluator=evaluator,
                    labels_enum=labels_enum,
                    val_batch_size=val_batch_size,
                    snms_window=comp_score_snms_window,
                    threshold=comp_score_threshold,
                    epoch_nr=epoch_nr,
                )
                if _is_main:
                    if comp["avg_final_score"] > best_eval_metric:
                        best_eval_metric = comp["avg_final_score"]
                        if save_as:
                            torch.save(raw_model.state_dict(), save_as)

                    summary_writer.add_scalar(
                        "eval/competition_score",
                        comp["avg_final_score"],
                        epoch_nr,
                    )
                    summary_writer.add_scalar(
                        "eval/competition_raw_score",
                        comp["avg_raw_score"],
                        epoch_nr,
                    )
                    for cls_name, pc in comp.get("per_class", {}).items():
                        summary_writer.add_scalar(
                            f"eval_comp_per_class_matched/{cls_name}",
                            pc.get("matched_points", 0.0),
                            epoch_nr,
                        )
                        summary_writer.add_scalar(
                            f"eval_comp_per_class_fp/{cls_name}",
                            pc.get("fp_penalty", 0.0),
                            epoch_nr,
                        )
            elif _is_main:
                if eval_metric == "loss":
                    eval_loss = _get_eval_loss(
                        model,
                        labels_enum=labels_enum,
                        data_loader=eval_data_loader,
                        foreground_weight=foreground_weight,
                        device=device,
                        per_class_weights=per_class_weights,
                        epoch_nr=epoch_nr,
                        summary_writer=summary_writer,
                        focal_loss_gamma=focal_loss_gamma,
                        autocast_dtype=autocast_dtype,
                    )
                    if eval_loss.total_loss < best_eval_metric:
                        best_eval_metric = eval_loss.total_loss
                        if save_as:
                            torch.save(raw_model.state_dict(), save_as)

                    summary_writer.add_scalar(
                        "eval/total_loss", eval_loss.total_loss, epoch_nr
                    )
                    summary_writer.add_scalar(
                        "eval/ce_labels_loss", eval_loss.ce_labels_loss, epoch_nr
                    )
                    summary_writer.add_scalar(
                        "eval/mse_displacement_loss",
                        eval_loss.mse_displacement_loss,
                        epoch_nr,
                    )
                elif eval_metric == "map":
                    maps, map_mine = evaluator.eval(batch_size=val_batch_size)
                    if map_mine > best_eval_metric:
                        best_eval_metric = map_mine
                        if save_as:
                            torch.save(raw_model.state_dict(), save_as)

                    summary_writer.add_scalar(
                        "eval/map_mine",
                        map_mine,
                        epoch_nr,
                    )

                    if maps.get("mAP") is not None:
                        summary_writer.add_scalar(
                            "eval/map_soccernet",
                            maps["mAP"],
                            epoch_nr,
                        )

            # All ranks must symmetrically return to train mode before the
            # next training epoch starts (otherwise BN/Dropout state diverges
            # across ranks).
            model.train()

            # Sync barrier so any straggler-rank doesn't immediately race into
            # the next train iteration's allreduce while the others are still
            # in eval. Cheap insurance even after the eval sharding fix.
            if _is_ddp:
                dist.barrier()

        if _is_main and save_as and save_every_epoch:
            epoch_path = f"{save_as}.epoch_{epoch_nr}.pt"
            torch.save(raw_model.state_dict(), epoch_path)
            print(f"Saved per-epoch checkpoint: {epoch_path}")
    return raw_model


def _get_eval_loss(
    model: TDeedModule,
    labels_enum: Type[BASLabel | ActionLabel],
    data_loader: DataLoader[TeamTDeedDataset],
    foreground_weight: int = 5,
    device=DEFAULT_DEVICE,
    per_class_weights=None,
    epoch_nr: int = None,
    summary_writer: SummaryWriter = None,
    focal_loss_gamma: float = 0.0,
    autocast_dtype: torch.dtype = torch.float16,
) -> TDeedLoss:
    epoch_loss_c, epoch_loss_d, ce_per_class = _go_through_epoch_eval(
        model,
        labels_enum=labels_enum,
        data_loader=data_loader,
        foreground_weight=foreground_weight,
        device=device,
        per_class_weights=per_class_weights,
        epoch_nr=epoch_nr,
        summary_writer=summary_writer,
        focal_loss_gamma=focal_loss_gamma,
        autocast_dtype=autocast_dtype,
    )
    epoch_loss = epoch_loss_c.detach().item()
    epoch_loss += epoch_loss_d.detach().item()

    return TDeedLoss(
        total_loss=epoch_loss / len(data_loader),
        ce_labels_loss=epoch_loss_c.detach().item() / len(data_loader),
        mse_displacement_loss=epoch_loss_d.detach().item() / len(data_loader),
        ce_per_class=ce_per_class,
    )


def _go_through_epoch_train(
    model: TDeedModule,
    labels_enum: Type[BASLabel | ActionLabel],
    data_loader: DataLoader[TeamTDeedDataset],
    scheduler: LRScheduler = None,
    scaler: torch.amp.GradScaler = None,
    optimizer: torch.optim.Optimizer = None,
    acc_grad_iter: int = None,
    epoch_nr: int = None,
    foreground_weight: int = 5,
    device=DEFAULT_DEVICE,
    summary_writer: SummaryWriter = None,
    loss_weights=None,
    is_main: bool = True,
    per_class_weights=None,
    focal_loss_gamma: float = 0.0,
    autocast_dtype: torch.dtype = torch.float16,
    max_iter: Optional[int] = None,
):
    return _go_through_epoch(
        model=model,
        labels_enum=labels_enum,
        data_loader=data_loader,
        evaluate=False,
        scheduler=scheduler,
        scaler=scaler,
        optimizer=optimizer,
        max_iter=max_iter,
        acc_grad_iter=acc_grad_iter,
        epoch_nr=epoch_nr,
        foreground_weight=foreground_weight,
        device=device,
        summary_writer=summary_writer,
        loss_weights=loss_weights or [1.5, 1],
        is_main=is_main,
        per_class_weights=per_class_weights,
        focal_loss_gamma=focal_loss_gamma,
        autocast_dtype=autocast_dtype,
    )


def _go_through_epoch_eval(
    model: TDeedModule,
    labels_enum: Type[BASLabel | ActionLabel],
    data_loader: DataLoader[TeamTDeedDataset],
    foreground_weight: int = 5,
    device: str = DEFAULT_DEVICE,
    loss_weights=None,
    per_class_weights=None,
    epoch_nr: int = None,
    summary_writer: SummaryWriter = None,
    focal_loss_gamma: float = 0.0,
    autocast_dtype: torch.dtype = torch.float16,
):

    return _go_through_epoch(
        model=model,
        labels_enum=labels_enum,
        data_loader=data_loader,
        evaluate=True,
        foreground_weight=foreground_weight,
        device=device,
        loss_weights=loss_weights or [1.5, 1],
        per_class_weights=per_class_weights,
        epoch_nr=epoch_nr,
        summary_writer=summary_writer,
        focal_loss_gamma=focal_loss_gamma,
        autocast_dtype=autocast_dtype,
    )


def _go_through_epoch(
    model: TDeedModule,
    labels_enum: Type[BASLabel | ActionLabel],
    data_loader: DataLoader[TeamTDeedDataset],
    evaluate: bool = False,
    scheduler: LRScheduler = None,
    scaler: torch.amp.GradScaler = None,
    optimizer: torch.optim.Optimizer = None,
    acc_grad_iter: int = None,
    epoch_nr: int = None,
    foreground_weight: float = 5,
    device=DEFAULT_DEVICE,
    summary_writer: SummaryWriter = None,
    loss_weights=None,
    is_main: bool = True,
    per_class_weights=None,
    focal_loss_gamma: float = 0.0,
    autocast_dtype: torch.dtype = torch.float16,
    max_iter: Optional[int] = None,
):
    loss_weights = loss_weights or [1.5, 1]
    if not evaluate:
        optimizer.zero_grad()

    epoch_loss_c = 0.0
    epoch_loss_d = 0.0

    n_classes = len(labels_enum) + 1
    class_names = ["background"] + [c.name for c in labels_enum]
    ce_per_class_sum = torch.zeros(n_classes, device=device)
    ce_per_class_cnt = torch.zeros(n_classes, device=device)
    pred_count = torch.zeros(n_classes, device=device)
    true_count = torch.zeros(n_classes, device=device)
    max_prob_sum = torch.zeros(1, device=device)
    max_prob_cnt = torch.zeros(1, device=device)

    if per_class_weights is not None:
        class_weights = torch.FloatTensor(
            [1.0] + [foreground_weight * w for w in per_class_weights]
        ).to(device)
    else:
        class_weights = torch.FloatTensor(
            [1] + [foreground_weight for _ in labels_enum]
        ).to(device)

    _amp_device_type = device.split(":")[0]

    batch_idx = 0
    with torch.no_grad() if evaluate else nullcontext():
        for batch in tqdm(data_loader, total=len(data_loader), disable=not is_main):

            clip_tensor = batch["clip_tensor"].to(device, non_blocking=True)
            label = batch["labels_vector"].to(device, non_blocking=True)
            labels_displacement_vector = batch["labels_displacement_vector"].to(
                device, non_blocking=True
            )

            label = (
                label.flatten()
                if len(label.shape) == 2
                else label.view(-1, label.shape[-1])
            )

            with torch.amp.autocast(device_type=_amp_device_type, dtype=autocast_dtype):
                pred_dict, y = model(clip_tensor, y=label, inference=evaluate)

                pred = pred_dict["im_feat"]

                if "displ_feat" in pred_dict.keys():
                    pred_displacement = pred_dict["displ_feat"]

                loss = 0.0
                loss_c = 0.0

                predictions = pred.reshape(-1, len(labels_enum) + 1)
                if focal_loss_gamma > 0.0:
                    # Focal loss: down-weight easy examples (high p_t).
                    # FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
                    # Keep per-class alpha via class_weights; reduction=mean over samples.
                    # label may be 1D class-idx or 2D one-hot/soft; normalise to 1D long.
                    label_idx_long = (
                        label.long() if label.dim() == 1 else label.argmax(dim=-1).long()
                    )
                    log_probs = F.log_softmax(predictions.float(), dim=-1)
                    log_p_t = log_probs.gather(
                        dim=-1, index=label_idx_long.unsqueeze(-1)
                    ).squeeze(-1)
                    p_t = log_p_t.exp().clamp(min=1e-8, max=1 - 1e-8)
                    focal_weight = (1.0 - p_t).pow(focal_loss_gamma)
                    alpha_t = class_weights.gather(dim=-1, index=label_idx_long)
                    per_sample = -(alpha_t * focal_weight * log_p_t)
                    loss_c = loss_c + per_sample.mean().to(predictions.dtype)
                else:
                    loss_c += F.cross_entropy(predictions, label, weight=class_weights)

                loss += loss_c * loss_weights[0]

                loss_d = F.mse_loss(
                    pred_displacement,
                    labels_displacement_vector,
                    reduction="none",
                )

                loss_d = loss_d.mean()
                loss += loss_d * loss_weights[1]

            epoch_loss_c += (loss_c * loss_weights[0]).detach()
            epoch_loss_d += (loss_d * loss_weights[1]).detach()

            with torch.no_grad():
                label_idx = label if label.dim() == 1 else label.argmax(dim=-1)
                ce_none = F.cross_entropy(
                    predictions.float(), label_idx, reduction="none"
                )
                probs = predictions.float().softmax(dim=-1)
                pred_idx = probs.argmax(dim=-1)
                max_prob = probs.max(dim=-1).values
                max_prob_sum += max_prob.sum()
                max_prob_cnt += max_prob.numel()
                for c in range(n_classes):
                    mask = label_idx == c
                    if mask.any():
                        ce_per_class_sum[c] += ce_none[mask].sum()
                        ce_per_class_cnt[c] += mask.sum()
                    true_count[c] += mask.sum()
                    pred_count[c] += (pred_idx == c).sum()

            if not evaluate:
                if is_main and summary_writer:
                    summary_writer.add_scalar(
                        "train/loss",
                        loss.detach().item(),
                        epoch_nr * len(data_loader) + batch_idx,
                    )
                    summary_writer.add_scalar(
                        "train/learning_rate",
                        optimizer.param_groups[0]["lr"],
                        epoch_nr * len(data_loader) + batch_idx,
                    )
                elif is_main:
                    print("train/loss", loss.detach().item())
            if not evaluate:
                _optim_step(
                    scaler=scaler,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss=loss,
                    backward_only=(batch_idx + 1) % acc_grad_iter != 0,
                )
            batch_idx += 1
            # Smoke-test break: cap an epoch to N iterations so eval can be
            # reached quickly. Used to verify multi-rank eval without waiting
            # 1.5h for a full epoch.
            if max_iter is not None and batch_idx >= max_iter:
                if is_main:
                    print(
                        f"[smoke-test] early break at iter {batch_idx} "
                        f"(max_iter={max_iter})"
                    )
                break

    ce_per_class = {}
    for c in range(n_classes):
        cnt = ce_per_class_cnt[c].item()
        if cnt > 0:
            ce_per_class[class_names[c]] = ce_per_class_sum[c].item() / cnt

    phase = "eval" if evaluate else "train"
    if is_main and summary_writer is not None and epoch_nr is not None:
        for name, val in ce_per_class.items():
            summary_writer.add_scalar(f"{phase}_per_class_ce/{name}", val, epoch_nr)

    if is_main and ce_per_class:
        header = f"[{phase} epoch {epoch_nr}] per-class CE (unweighted):"
        parts = [f"{k}={v:.3f}" for k, v in ce_per_class.items()]
        print(header + " " + "  ".join(parts))

    total_true = true_count.sum().item()
    total_pred = pred_count.sum().item()
    if is_main and total_pred > 0:
        avg_confidence = (max_prob_sum / max_prob_cnt.clamp(min=1)).item()
        print(f"[{phase} epoch {epoch_nr}] prediction distribution (argmax):")
        for c in range(n_classes):
            tc = true_count[c].item()
            pc = pred_count[c].item()
            true_pct = 100 * tc / total_true if total_true > 0 else 0
            pred_pct = 100 * pc / total_pred if total_pred > 0 else 0
            print(
                f"    {class_names[c]:<20s} true={tc:>8.0f} ({true_pct:5.2f}%)  "
                f"pred={pc:>8.0f} ({pred_pct:5.2f}%)"
            )
        print(f"    avg_max_prob(confidence) = {avg_confidence:.3f}")
        if summary_writer is not None and epoch_nr is not None:
            summary_writer.add_scalar(
                f"{phase}_avg_confidence", avg_confidence, epoch_nr
            )
            for c in range(n_classes):
                summary_writer.add_scalar(
                    f"{phase}_pred_frac/{class_names[c]}",
                    pred_count[c].item() / max(total_pred, 1),
                    epoch_nr,
                )

    return epoch_loss_c, epoch_loss_d, ce_per_class


def _eval_competition_score(
    evaluator: BASTeamTDeedEvaluator,
    labels_enum: Type[BASLabel | ActionLabel],
    val_batch_size: int,
    snms_window: int = 50,
    threshold: float = 0.5,
    epoch_nr: Optional[int] = None,
) -> dict:
    """Run full-video inference, convert scores -> events using fixed SNMS + threshold,
    then score each video against its Labels-ball.json ground truth and return the
    aggregate competition score.

    Under DDP, each rank handles a disjoint subset of videos. Per-video results
    are gathered to all ranks, and rank 0 produces the final aggregate prints.

    Returns dict with: avg_final_score, avg_raw_score, per_class {matched_points, fp_penalty}.
    """
    import numpy as np

    is_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_ddp else 0
    world_size = dist.get_world_size() if is_ddp else 1

    # Each rank only runs inference on its sharded subset of videos.
    scored_videos = evaluator.get_scored_videos(
        batch_size=val_batch_size,
        use_snms=False,
        use_hflip=False,
        snms_params=None,
        rank=rank,
        world_size=world_size,
    )

    class_list = list(labels_enum)
    per_video_results: list = []
    per_class_agg: dict = {}

    for vd in scored_videos:
        scores = vd.scores  # (n_frames, n_classes)
        # SNMS turns raw per-frame scores into sparse "peak" scores (non-peak frames = 0)
        post = soft_non_maximum_suppression(
            scores, class_window=snms_window, threshold=0.01
        )
        fps = float(vd.video.metadata_fps)

        pred_events: list[Event] = []
        for class_idx, cls in enumerate(class_list):
            col = post[:, class_idx]
            frames = np.where(col > threshold)[0]
            for f in frames:
                pred_events.append(
                    Event(
                        label=cls.value,
                        time_ms=int(f / fps * 1000),
                        frame_nr=int(f),
                        confidence=float(col[f]),
                    )
                )

        gt_path = os.path.join(
            os.path.dirname(vd.video.absolute_path), "Labels-ball.json"
        )
        gt_events = load_all_gt_events(gt_path)
        result = score_video(gt_events, pred_events)
        per_video_results.append(result)

        for cls_name, pc in result.per_class.items():
            agg = per_class_agg.setdefault(
                cls_name, {"matched_points": 0.0, "fp_penalty": 0.0, "n_matched": 0, "n_fp": 0, "n_gt": 0}
            )
            for k in ("matched_points", "fp_penalty", "n_matched", "n_fp", "n_gt"):
                agg[k] += pc.get(k, 0)

    # Gather per-rank results onto every rank, then rebuild the global aggregate.
    # We use all_gather_object so plain Python objects (results, dicts) can be
    # transmitted without manual tensor-encoding. This is fine for ~tens of
    # videos per epoch; cost is ~milliseconds.
    if is_ddp and world_size > 1:
        gathered_per_video: list = [None] * world_size
        gathered_per_class: list = [None] * world_size
        dist.all_gather_object(gathered_per_video, per_video_results)
        dist.all_gather_object(gathered_per_class, per_class_agg)

        # Flatten per-video lists in rank order so video order is deterministic.
        per_video_results = [r for rank_list in gathered_per_video for r in rank_list]

        # Combine per-class dicts by summing across ranks.
        merged_per_class: dict = {}
        for d in gathered_per_class:
            for cls_name, pc in d.items():
                agg = merged_per_class.setdefault(
                    cls_name, {"matched_points": 0.0, "fp_penalty": 0.0, "n_matched": 0, "n_fp": 0, "n_gt": 0}
                )
                for k in ("matched_points", "fp_penalty", "n_matched", "n_fp", "n_gt"):
                    agg[k] += pc.get(k, 0)
        per_class_agg = merged_per_class

    agg_final = aggregate_final_scores(per_video_results)
    if epoch_nr is not None and rank == 0:
        print(
            f"[eval epoch {epoch_nr}] competition_score "
            f"(snms_window={snms_window}, threshold={threshold:.2f}): "
            f"avg_final={agg_final['avg_final_score']:.4f}  "
            f"avg_raw={agg_final['avg_raw_score']:.4f}  "
            f"matched_sum={agg_final['sum_matched']:.1f}  "
            f"fp_sum={agg_final['sum_fp_penalty']:.1f}  "
            f"gt_weight_sum={agg_final['sum_gt_weight']:.1f}"
        )
        if per_class_agg:
            print(f"[eval epoch {epoch_nr}] competition per-class:")
            for cls in labels_enum:
                pc = per_class_agg.get(cls.value, {})
                m = pc.get("matched_points", 0.0)
                fp = pc.get("fp_penalty", 0.0)
                n_gt = int(pc.get("n_gt", 0))
                n_m = int(pc.get("n_matched", 0))
                n_fp = int(pc.get("n_fp", 0))
                contribution = m - fp
                print(
                    f"    {cls.name:<20s} match={n_m}/{n_gt}  fp={n_fp}  "
                    f"+pts={m:6.2f}  -pen={fp:6.2f}  net={contribution:+7.2f}"
                )

    return {
        "avg_final_score": agg_final["avg_final_score"],
        "avg_raw_score": agg_final["avg_raw_score"],
        "per_class": per_class_agg,
    }


def _optim_step(scaler, optimizer, scheduler, loss, backward_only=False):
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()

    if not backward_only:
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
