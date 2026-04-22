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

# Per-class SNMS windows. Order MUST match Action enum (11 classes, CLEARANCE dropped):
#   PASS, PASS_RECEIVED, RECOVERY, TACKLE, INTERCEPTION,
#   BALL_OUT_OF_PLAY, AERIAL_DUEL, SHOT, SAVE, FOUL, GOAL
# Baseline derived from w=12/25/50/75 sweep on competition_videos_val:
#   - tight (50) for frequent / low-tolerance events (PASS family, TACKLE, SAVE)
#   - wider (75) for sparse / high-tolerance events (BALL_OUT, GOAL)
uv run python -u dudek/scripts/tdeed.py evaluate-competition \
    --val_dataset_path=/workspace/bas/data/competition_videos_val/ \
    --model_checkpoint_path=/workspace/bas/bt-soccer-bas2/tdeed_competition_640_4.pt \
    --clip_frames_count=170 \
    --overlap=136 \
    --val_batch_size=2 \
    --use_snms=true \
    --snms_windows="50,50,50,50,50,75,50,50,50,50,75" \
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