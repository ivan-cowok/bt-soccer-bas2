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


##### v3.5 baseline: v3 + augmentation cleanup
#
# Augmentation changes vs v3:
#   - camera_move_proba: 0.2 -> 0.0
#       Real cameras in our footage are mostly fixed (slow human-pan only when
#       ball goes long). The current sinusoidal warp augmentation injects
#       motion that does not occur at test time and adds zero-padding
#       artifacts at frame edges. Removing it.
#   - gaussian_blur_kernel_size: 5 -> 3
#       Plus the model code now caps GaussianBlur sigma at (0.3, 1.0) (was the
#       torchvision default (0.1, 2.0) which is destructive on a 6-10 px ball
#       at 640x360).
#   - NEW: AddGaussianNoise (per-frame independent noise, sigma per clip
#       0.0 - 0.03 in [0,1] units, p=0.20). Simulates sensor / snow / low-quality
#       grain present in our footage. Implemented in
#       dudek/ml/model/tdeed/modules/tdeed.py and wired into the model's
#       augmentation pipeline (no CLI knob; tweak the source if needed).
#   - NEW: RandomErasing (same box across all frames in a clip via v2's per-
#       call param sampling, scale=(0.005, 0.03), p=0.15, value=0). Simulates
#       player / observer / object occlusions. Forces the model to use temporal
#       context and player-action cues instead of ball-only shortcuts. Expected
#       to most help PASS_RECEIVED / AERIAL_DUEL / SHOT FP rates (the classes
#       currently over-relying on ball-pixel patterns). No CLI knob.
#   - ColorJitter ranges tuned to our domain (no CLI knobs):
#         hue:        0.10        -> 0.05         (tighter; ±10% hue rotation
#                                                 made grass purple-ish.)
#         saturation: (0.7, 1.2)  -> (0.5, 1.3)   (wider; snow matches need
#                                                 low end, daylight high end.)
#         brightness: (0.7, 1.2)  -> (0.6, 1.3)   (wider; cover snow + evening.)
#         contrast:   (0.7, 1.2)  -> (0.8, 1.15)  (tighter; low contrast on
#                                                 cheap footage erases ball-
#                                                 vs-grass detail.)
#   - flip_proba: 0.3 -> 0.5
#       Soccer is left/right symmetric for our task (PASS/PASS_RECEIVED/etc.
#       look identical mirrored). 0.5 is the standard for symmetric vision
#       tasks; 0.3 was the unexamined inherited choice. Note: the dataset
#       code (dudek/ml/data/tdeed.py) was also patched so flip is now clamped
#       at 0.5 inside the even_choice doubling block (otherwise rare-event
#       clips would always be flipped, losing orientation diversity).
#   - displacement: 4 -> 3
#       The ±4 edge frames are visually the most ambiguous (foot approaching
#       or leaving ball, ball mid-flight) and contribute to the model firing
#       PASS-family classes liberally on neutral-looking frames at inference.
#       Tightening to ±3 removes those edge frames. With our gap=2 minimum
#       between adjacent events, each event in an adjacent pair still gets
#       4-5 labeled frames (vs 5-6 at d=4) — plenty of training signal given
#       1000s of events per common class. Expected modest FP reduction on
#       PASS / PASS_RECEIVED.
#
# Everything else identical to v3 so we can attribute changes to the
# augmentation cleanup alone.
mkdir -p /workspace/bas/logs && \
LOG_FILE=/workspace/bas/logs/train_focal_v35_$(date +%Y%m%d_%H%M%S).log && \
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
    --flip_proba=0.5 \
    --crop_proba=0.2 \
    --camera_move_proba=0.0 \
    --even_choice_proba=0.4 \
    --loss_foreground_weight=5 \
    --backbone_lr_scale=0.1 \
    --weight_decay=0.05 \
    --class_weight_mode=inverse_sqrt \
    --class_weight_cap=3.0 \
    --grad_checkpointing=true \
    --gaussian_blur_kernel_size=3 \
    --displacement=3 \
    --focal_loss_gamma=1.0 \
    --eval_metric=competition_score \
    --comp_score_snms_window=50 \
    --comp_score_threshold=0.30 \
    --start_eval_epoch_nr=2 \
    --save_every_epoch=true \
    --save_as=tdeed_competition_focal_v35.pt 2>&1 | tee "$LOG_FILE"


##### v3: post-training per-class threshold sweep — single best checkpoint
#
# After full v3 training (15 epochs), epoch 12 is the clear winner:
#   - avg_final at theta=0.30 = 0.0710 (33%+ above any other epoch).
#   - Best per-class score on 6 of 8 classes: RECOVERY, BALL_OUT_OF_PLAY,
#     SHOT, SAVE, GOAL, PASS_RECEIVED. PASS and AERIAL_DUEL prefer later
#     epochs but their gains are dwarfed by rare-class regressions.
#   - Lowest fp_sum of all epochs (907 vs 972-1078 elsewhere). Better
#     starting precision -> better post-sweep score.
#   - Epochs 13-15 show the classic late-training FP-creep pattern: train
#     CE keeps dropping but validation FPs grow faster than matches as the
#     model overgeneralises on rare classes.
uv run python -u dudek/scripts/tdeed.py evaluate-competition \
    --val_dataset_path=/workspace/bas/data/competition_videos_val/ \
    --model_checkpoint_path=/workspace/bas/bt-soccer-bas2/tdeed_competition_focal_v3.pt.epoch_12.pt \
    --clip_frames_count=170 \
    --overlap=136 \
    --val_batch_size=2 \
    --optimize_thresholds \
    --threshold_sweep="0.05,0.95,0.05" \
    --snms_window_sweep="25,50,75,100" \
    --output_dir=/workspace/bas/sweep/v3_epoch_12/


# =============================================================================
# v4 baseline: 960x540 input + Sequence Transformer head + bf16
#
# Three changes from v3 / v3.5:
#
# 1) Resolution 640x360 -> 960x540 (2.25x more pixels per frame).
#    Motivation: the PASS / PASS_RECEIVED / RECOVERY trio sits at the visual
#    ambiguity floor at 360p — the 80ms (gap=2) gap between adjacent PASS and
#    PASS_RECEIVED is at the edge of pixel-level distinguishability when the
#    foot/ball region is ~6-10 px. 540p ~doubles the linear pixel budget on
#    the foot/ball interaction, which is where the per-class CE for those
#    three classes is bottlenecked.
#
#    Memory: at batch=1 per GPU + grad_checkpointing + bf16, expected ~22GB
#    on a 32GB 5090 with the new attention head. With 4x 5090 DDP and
#    acc_grad_iter=2 we get an effective batch of 8, matching v3/v3.5.
#
# 2) Stacked SequenceTransformerHead (2 layers, 8 heads, pre-LN, learned PE)
#    inserted between SGP and the per-frame heads. SGP keeps capturing
#    short-range temporal structure (1-15 frames); the new head adds full-
#    clip attention so each frame can pull evidence from up to 169 other
#    frames. Specifically targets:
#      - PASS_RECEIVED needing a recent PASS (5-25 frames back),
#      - RECOVERY's diverse predecessors (5-50 frames: SHOT / AERIAL_DUEL /
#        failed PASS / BOOP aftermath),
#      - Stoppage suppression after BALL_OUT_OF_PLAY.
#    The Transformer is stacked, NOT replacing SGP, to preserve TDEED's BAS-
#    pretrained `_temp_fine` weights from tdeed_best.pt.
#
# 3) bf16 mixed precision (was fp16). Same speed on Blackwell, fp32-equivalent
#    dynamic range, no GradScaler needed. Critical for attention/softmax
#    numerical stability — fp16 underflow corrupts gradients in the new
#    Transformer head, bf16 does not.
#
# Warm-start strategy:
#   - Backbone + SGP: warm-start from tdeed_best.pt (BAS-finetuned).
#     `load_backbone()` only loads `_features` + `_temp_fine`; the new
#     `_seq_decoder` and the per-frame heads are randomly initialised.
#   - --seq_freeze_epochs=2: freeze backbone + SGP for the first 2 epochs so
#     the random Transformer head settles before its gradients flow back
#     through the BAS-pretrained weights. Optimizer is rebuilt at epoch 2
#     to bring backbone + SGP back in (with `backbone_lr_scale=0.1`).
#
# Backbone resolution mismatch note:
#   tdeed_best.pt was BAS-trained at lower resolution. Going to 540p does
#   change the relative scale of features, but the CNN is mostly scale-
#   translation invariant; the first 2 epochs (with backbone frozen) give
#   the head time to settle, then `backbone_lr_scale=0.1` keeps adaptation
#   gentle.
#
# Hardware: 4x RTX 5090 with DDP via torchrun.
# Expected wall-clock: ~1.5-2 h/epoch, 14 epochs ~24-30 h.
# =============================================================================


# -------- Step 1: re-extract competition + custom frames at 960x540 --------
uv run python dudek/scripts/extract.py extract-competition-frames \
    --dataset_path=/workspace/bas/data/competition_videos/ \
    --resolution=224 \
    --stride=1 \
    --frame_target_width=960 \
    --frame_target_height=540 \
    --num_workers=10

uv run python dudek/scripts/extract.py extract-competition-frames \
    --dataset_path=/workspace/bas/data/competition_videos_val/ \
    --resolution=224 \
    --stride=1 \
    --frame_target_width=960 \
    --frame_target_height=540 \
    --num_workers=8


# -------- Step 2: train v4 on 4x RTX 5090 via DDP --------
mkdir -p /workspace/bas/logs && \
LOG_FILE=/workspace/bas/logs/train_v4_$(date +%Y%m%d_%H%M%S).log && \
echo "Logging to: $LOG_FILE" && \
torchrun --nproc_per_node=4 dudek/scripts/tdeed.py train-competition \
    --dataset_path=/workspace/bas/data/competition_videos/ \
    --val_dataset_path=/workspace/bas/data/competition_videos_val/ \
    --model_checkpoint_path=/workspace/bas/bt-soccer-bas2/tdeed_best.pt \
    --clip_frames_count=170 \
    --overlap=136 \
    --nr_epochs=20 \
    --learning_rate=0.00005 \
    --train_batch_size=1 \
    --val_batch_size=1 \
    --acc_grad_iter=2 \
    --num_workers=6 \
    --flip_proba=0.5 \
    --crop_proba=0.2 \
    --camera_move_proba=0.0 \
    --even_choice_proba=0.4 \
    --loss_foreground_weight=5 \
    --backbone_lr_scale=0.1 \
    --weight_decay=0.05 \
    --class_weight_mode=inverse_sqrt \
    --class_weight_cap=3.0 \
    --grad_checkpointing=true \
    --gaussian_blur_kernel_size=3 \
    --displacement=3 \
    --focal_loss_gamma=1.0 \
    --eval_metric=competition_score \
    --comp_score_snms_window=50 \
    --comp_score_threshold=0.30 \
    --start_eval_epoch_nr=3 \
    --save_every_epoch=true \
    --use_bf16=true \
    --use_seq_decoder=true \
    --seq_decoder_layers=2 \
    --seq_decoder_heads=8 \
    --seq_decoder_ff_mult=4 \
    --seq_decoder_dropout=0.1 \
    --seq_freeze_epochs=2 \
    --save_as=tdeed_competition_v4.pt 2>&1 | tee "$LOG_FILE"


# -------- Step 3: post-training per-epoch sweep (v4) --------
# v4 checkpoints have `_seq_decoder.*` keys, so evaluate-competition must be
# called with --use_seq_decoder=true and the matching seq_decoder_* params.
for CKPT in /workspace/bas/bt-soccer-bas2/tdeed_competition_v4.pt.epoch_*.pt; do
    EPOCH=$(echo "$CKPT" | grep -oP 'epoch_\K[0-9]+')
    uv run python -u dudek/scripts/tdeed.py evaluate-competition \
        --val_dataset_path=/workspace/bas/data/competition_videos_val/ \
        --model_checkpoint_path="$CKPT" \
        --clip_frames_count=170 \
        --overlap=136 \
        --val_batch_size=1 \
        --use_seq_decoder=true \
        --seq_decoder_layers=2 \
        --seq_decoder_heads=8 \
        --seq_decoder_ff_mult=4 \
        --seq_decoder_dropout=0.1 \
        --optimize_thresholds \
        --threshold_sweep="0.05,0.95,0.05" \
        --snms_window_sweep="25,50,75,100" \
        --output_dir="/workspace/bas/sweep/v4_epoch_${EPOCH}/"
done