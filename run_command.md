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

uv run python dudek/scripts/tdeed.py train-competition \
    --dataset_path=/workspace/bas/data/competition_videos/ \
    --val_dataset_path=/workspace/bas/data/competition_videos_val/ \
    --model_checkpoint_path=/workspace/bas/bt-soccer-bas2/pretrained.pt \
    --clip_frames_count=140 \
    --overlap=112 \
    --nr_epochs=30 \
    --learning_rate=0.00001 \
    --train_batch_size=2 \
    --val_batch_size=2 \
    --acc_grad_iter=4 \
    --num_workers=8 \
    --flip_proba=0.5 \
    --crop_proba=0.4 \
    --camera_move_proba=0.4 \
    --even_choice_proba=0.5 \
    --loss_foreground_weight=5 \
    --backbone_lr_scale=0.1 \
    --weight_decay=0.01 \
    --class_weight_mode=none \
    --class_weight_cap=3.0 \
    --grad_checkpointing=true \
    --save_as=tdeed_competition_640_2.pt