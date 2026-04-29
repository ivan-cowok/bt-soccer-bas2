# =============================================================================
# NEW 12-CLASS SCHEMA (clearance removed, block added)
#   Action enum order:
#   PASS, PASS_RECEIVED, RECOVERY, TACKLE, INTERCEPTION,
#   BALL_OUT_OF_PLAY, BLOCK, AERIAL_DUEL, SHOT, SAVE, FOUL, GOAL
#
# All videos are now 25fps and labels match. Finetune from tdeed_best.pt
# (BAS-finetuned checkpoint, 7-class head — head is re-initialised to 12
# classes by load_backbone() which only loads _features + _temp_fine).
# =============================================================================


# -------- Step 1: extract 640x360 frames (stride=1, all frames) --------
#
# NOTE on --resolution: this is the filename suffix of the source video
# ({resolution}p.mp4, e.g. 224p.mp4). It is NOT pixel size. The actual
# extracted JPG size is controlled by --frame_target_width/--frame_target_height.
# Keep --resolution matching what your video files are named on disk
# (e.g. 224p.mp4 -> --resolution=224).

uv run python dudek/scripts/extract.py extract-competition-frames \
    --dataset_path=/workspace/bas/data/competition_videos/ \
    --resolution=224 \
    --stride=1 \
    --frame_target_width=640 \
    --frame_target_height=360 \
    --num_workers=10

uv run python dudek/scripts/extract.py extract-competition-frames \
    --dataset_path=/workspace/bas/data/competition_videos_val/ \
    --resolution=224 \
    --stride=1 \
    --frame_target_width=640 \
    --frame_target_height=360 \
    --num_workers=8


# -------- Step 2: finetune from tdeed_best.pt with Focal Loss + competition_score --------
#
# Key changes vs prior run:
#   --eval_metric=competition_score  → saves best checkpoint by avg clamped
#       competition score (FP-penalty aware), not CE loss or mAP.
#   --focal_loss_gamma=2.0           → focuses training on hard examples,
#       helps rare classes (GOAL, FOUL, SAVE, BLOCK, AERIAL_DUEL).
#   --comp_score_threshold=0.5       → fixed threshold used only to compute
#       the per-epoch competition score for checkpoint selection. Real
#       per-class optimal thresholds are found post-training via
#       evaluate-competition --optimize_thresholds.
#   --comp_score_snms_window=50      → single fixed SNMS window for the same.
#
# class_weight_cap=2.0 still applies with inverse_sqrt; combined with focal
# loss this gives a strong but bounded push on rare classes.

mkdir -p /workspace/bas/logs && \
LOG_FILE=/workspace/bas/logs/train_focal_$(date +%Y%m%d_%H%M%S).log && \
echo "Logging to: $LOG_FILE" && \
uv run python -u dudek/scripts/tdeed.py train-competition \
    --dataset_path=/workspace/bas/data/competition_videos/ \
    --val_dataset_path=/workspace/bas/data/competition_videos_val/ \
    --model_checkpoint_path=/workspace/bas/bt-soccer-bas2/tdeed_best.pt \
    --clip_frames_count=170 \
    --overlap=136 \
    --nr_epochs=30 \
    --learning_rate=0.00005 \
    --train_batch_size=2 \
    --val_batch_size=2 \
    --acc_grad_iter=4 \
    --num_workers=12 \
    --flip_proba=0.3 \
    --crop_proba=0.2 \
    --camera_move_proba=0.2 \
    --even_choice_proba=0.2 \
    --loss_foreground_weight=5 \
    --backbone_lr_scale=0.1 \
    --weight_decay=0.05 \
    --class_weight_mode=inverse_sqrt \
    --class_weight_cap=2.0 \
    --grad_checkpointing=true \
    --focal_loss_gamma=2.0 \
    --eval_metric=competition_score \
    --start_eval_epoch_nr=3 \
    --comp_score_snms_window=50 \
    --comp_score_threshold=0.5 \
    --save_as=tdeed_competition_focal_1.pt 2>&1 | tee "$LOG_FILE"


# -------- Step 3: post-training per-class threshold & SNMS-window sweep --------
# Uses the full 15-class GT weight for the denominator (matches real scoring).

uv run python -u dudek/scripts/tdeed.py evaluate-competition \
    --val_dataset_path=/workspace/bas/data/competition_videos_val/ \
    --model_checkpoint_path=/workspace/bas/bt-soccer-bas2/tdeed_competition_focal_1.pt \
    --clip_frames_count=170 \
    --overlap=136 \
    --val_batch_size=2 \
    --optimize_thresholds \
    --threshold_sweep="0.05,0.95,0.05" \
    --snms_window_sweep="25,50,75,100" \
    --output_dir=/workspace/bas/sweep/focal_opt/


# =============================================================================
# LEGACY COMMANDS (kept for reference, 11-class schema, pre-fix data)
# =============================================================================


torchrun --nproc_per_node=2 dudek/scripts/tdeed.py train \
    --dataset_path=/workspace/bas/data/bas_videos/ \
    --model_checkpoint_path=/workspace/bas/bt-soccer-bas2/pretrained.pt \
    --resolution=224 \
    --clip_frames_count=80 \
    --overlap=68 \
    --nr_epochs=40 \
    --train_batch_size=8 \
    --val_batch_size=8 \
    --enforce_train_epoch_size=6000 \
    --eval_metric=map \
    --start_eval_epoch_nr=5 \
    --save_as=tdeed_best.pt


torchrun --nproc_per_node=4 dudek/scripts/tdeed.py train-competition \
    --dataset_path=/workspace/bas/data/competition_videos/ \
    --model_checkpoint_path=/workspace/bas/bt-soccer-bas2/pretrained.pt \
    --resolution=224 \
    --clip_frames_count=80 \
    --overlap=68 \
    --nr_epochs=40 \
    --save_as=tdeed_competition.pt


uv run python dudek/scripts/tdeed.py train-competition \
    --dataset_path=/workspace/bas/data/competition_videos/ \
    --model_checkpoint_path=/workspace/bas/bt-soccer-bas2/pretrained.pt \
    --clip_frames_count=160 \
    --overlap=136 \
    --nr_epochs=40 \
    --save_as=tdeed_competition.pt


uv run bas-frame-extract extract-bas-frames \
    --dataset_path=/workspace/bas/data/competition_videos/ \
    --resolution=224 \
    --save_all=true \
    --stride=1 \
    --frame_target_width=640 \
    --frame_target_height=360

    mkdir -p /workspace/bas/logs && \
    LOG_FILE=/workspace/bas/logs/train_$(date +%Y%m%d_%H%M%S).log && \
    echo "Logging to: $LOG_FILE" && \
    uv run python -u dudek/scripts/tdeed.py train-competition \
        --dataset_path=/workspace/bas/data/competition_videos/ \
        --val_dataset_path=/workspace/bas/data/competition_videos_val/ \
        --model_checkpoint_path=/workspace/bas/bt-soccer-bas2/pretrained.pt \
        --clip_frames_count=170 \
        --overlap=136 \
        --nr_epochs=30 \
        --learning_rate=0.00005 \
        --train_batch_size=2 \
        --val_batch_size=2 \
        --acc_grad_iter=4 \
        --num_workers=12 \
        --flip_proba=0.3 \
        --crop_proba=0.2 \
        --camera_move_proba=0.2 \
        --even_choice_proba=0.2 \
        --loss_foreground_weight=5 \
        --backbone_lr_scale=0.1 \
        --weight_decay=0.05 \
        --class_weight_mode=inverse_sqrt \
        --class_weight_cap=2.0 \
        --grad_checkpointing=true \
        --save_as=tdeed_competition_640_4.pt 2>&1 | tee "$LOG_FILE"


    uv run python -u dudek/scripts/tdeed.py evaluate-competition \
        --val_dataset_path=/workspace/bas/data/competition_videos_val/ \
        --model_checkpoint_path=/workspace/bas/bt-soccer-bas2/tdeed_competition_640_4.pt \
        --clip_frames_count=170 \
        --overlap=136 \
        --val_batch_size=2 \
        --use_snms=true \
        --output_dir=/workspace/bas/eval_results/

uv run python -u dudek/scripts/tdeed.py evaluate-competition \
    --val_dataset_path=/workspace/bas/data/competition_videos_challenge/ \
    --model_checkpoint_path=/workspace/bas/bt-soccer-bas2/tdeed_competition_640_4.pt \
    --clip_frames_count=170 \
    --overlap=136 \
    --val_batch_size=2 \
    --use_snms=true \
    --output_dir=/workspace/bas/eval_results/ \
    --min_confidence=0.3

# Per-class SNMS windows. Order MUST match Action enum (8 classes, v3 schema):
#   PASS, PASS_RECEIVED, RECOVERY, BALL_OUT_OF_PLAY, AERIAL_DUEL, SHOT, SAVE, GOAL
# Baseline:
#   - tight (50) for frequent / low-tolerance events (PASS family)
#   - wider (75) for sparse / high-tolerance events (BALL_OUT_OF_PLAY, GOAL)
uv run python -u dudek/scripts/tdeed.py evaluate-competition \
    --val_dataset_path=/workspace/bas/data/competition_videos_val/ \
    --model_checkpoint_path=/workspace/bas/bt-soccer-bas2/tdeed_competition_focal_v3.pt \
    --clip_frames_count=170 \
    --overlap=136 \
    --val_batch_size=2 \
    --use_snms=true \
    --snms_windows="50,50,50,75,50,50,50,75" \
    --output_dir=/workspace/bas/sweep/per_class/


##### fine tune on finetune

mkdir -p /workspace/bas/logs && \
LOG_FILE=/workspace/bas/logs/train_basinit_$(date +%Y%m%d_%H%M%S).log && \
echo "Logging to: $LOG_FILE" && \
uv run python -u dudek/scripts/tdeed.py train-competition \
    --dataset_path=/workspace/bas/data/competition_videos/ \
    --val_dataset_path=/workspace/bas/data/competition_videos_val/ \
    --model_checkpoint_path=/workspace/bas/bt-soccer-bas2/tdeed_best.pt \
    --clip_frames_count=170 \
    --overlap=136 \
    --nr_epochs=30 \
    --learning_rate=0.00005 \
    --train_batch_size=2 \
    --val_batch_size=2 \
    --acc_grad_iter=4 \
    --num_workers=12 \
    --flip_proba=0.3 \
    --crop_proba=0.2 \
    --camera_move_proba=0.2 \
    --even_choice_proba=0.2 \
    --loss_foreground_weight=5 \
    --backbone_lr_scale=0.1 \
    --weight_decay=0.05 \
    --class_weight_mode=inverse_sqrt \
    --class_weight_cap=2.0 \
    --grad_checkpointing=true \
    --save_as=tdeed_competition_basinit_1.pt 2>&1 | tee "$LOG_FILE"


##### Optimize per-class (SNMS window, confidence threshold) vs competition score
#
# Runs inference WITHOUT pre-applied SNMS, then sweeps:
#   - SNMS window in {25, 50, 75, 100}
#   - confidence threshold in [0.05, 0.95] step 0.05
# per class, picking the combo that maximises total class contribution
# (matched_points - fp_penalty) across val videos under the score.py formula.
#
# Output: per-class optimal (window, threshold), TP/FP confidence percentiles,
# per-video raw/final scores, and the overall avg clamped final score (the
# competition metric). Classes with non-positive best contribution are flagged
# DROP (not submitted).
#
# Uses the full 15-class GT (including unsubmitted classes like clearance) for
# the denominator, matching the real scoring behaviour.

uv run python -u dudek/scripts/tdeed.py evaluate-competition \
    --val_dataset_path=/workspace/bas/data/competition_videos_val/ \
    --model_checkpoint_path=/workspace/bas/bt-soccer-bas2/tdeed_competition_640_4.pt \
    --clip_frames_count=170 \
    --overlap=136 \
    --val_batch_size=2 \
    --optimize_thresholds \
    --threshold_sweep="0.05,0.95,0.05" \
    --snms_window_sweep="25,50,75,100" \
    --output_dir=/workspace/bas/sweep/competition_opt/


##### v3 baseline: 8-class schema + focal γ=1.0 + per-epoch save
#
# Schema (Action enum, 8 classes, in order):
#   PASS, PASS_RECEIVED, RECOVERY, BALL_OUT_OF_PLAY,
#   AERIAL_DUEL, SHOT, SAVE, GOAL
# (TACKLE / INTERCEPTION / BLOCK / FOUL / CLEARANCE filtered out from data side.)
#
# Warm-start: tdeed_best.pt (BAS-finetuned on SoccerNet broadcast).
# Reason vs focal_v2.pt:
#   - focal_v2 backbone was shaped by gradients from the 4 now-dropped classes
#     on the older / smaller dataset. Carrying that bias into v3 is suboptimal.
#   - tdeed_best.pt was never specialized on those classes; it only provides
#     a clean, broadly-trained backbone. Head is dropped either way (different
#     class count) via load_backbone(), so we keep only `_features` + `_temp_fine`.
#
# `--save_every_epoch=true` keeps every epoch as
#   tdeed_competition_focal_v3.pt.epoch_{N}.pt
# in addition to the best one, so we can sweep all checkpoints later.
mkdir -p /workspace/bas/logs && \
LOG_FILE=/workspace/bas/logs/train_focal_v3_$(date +%Y%m%d_%H%M%S).log && \
echo "Logging to: $LOG_FILE" && \
uv run python -u dudek/scripts/tdeed.py train-competition \
    --dataset_path=/workspace/bas/data/competition_videos/ \
    --val_dataset_path=/workspace/bas/data/competition_videos_val/ \
    --model_checkpoint_path=/workspace/bas/bt-soccer-bas2/tdeed_best.pt \
    --clip_frames_count=170 \
    --overlap=136 \
    --nr_epochs=25 \
    --learning_rate=0.00005 \
    --train_batch_size=2 \
    --val_batch_size=2 \
    --acc_grad_iter=4 \
    --num_workers=12 \
    --flip_proba=0.3 \
    --crop_proba=0.2 \
    --camera_move_proba=0.2 \
    --even_choice_proba=0.4 \
    --loss_foreground_weight=5 \
    --backbone_lr_scale=0.1 \
    --weight_decay=0.05 \
    --class_weight_mode=inverse_sqrt \
    --class_weight_cap=3.0 \
    --grad_checkpointing=true \
    --focal_loss_gamma=1.0 \
    --eval_metric=competition_score \
    --comp_score_snms_window=50 \
    --comp_score_threshold=0.30 \
    --start_eval_epoch_nr=2 \
    --save_every_epoch=true \
    --save_as=tdeed_competition_focal_v3.pt 2>&1 | tee "$LOG_FILE"


##### v3: post-training per-class threshold sweep (run on all epoch ckpts)
# Loop epoch checkpoints and pick the best per the optimize_thresholds output.
for CKPT in /workspace/bas/bt-soccer-bas2/tdeed_competition_focal_v3.pt.epoch_*.pt; do
    EPOCH=$(echo "$CKPT" | grep -oP 'epoch_\K[0-9]+')
    uv run python -u dudek/scripts/tdeed.py evaluate-competition \
        --val_dataset_path=/workspace/bas/data/competition_videos_val/ \
        --model_checkpoint_path="$CKPT" \
        --clip_frames_count=170 \
        --overlap=136 \
        --val_batch_size=2 \
        --optimize_thresholds \
        --threshold_sweep="0.05,0.95,0.05" \
        --snms_window_sweep="25,50,75,100" \
        --output_dir="/workspace/bas/sweep/v3_epoch_${EPOCH}/"
done