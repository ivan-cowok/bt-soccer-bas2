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


 uv run torchrun --nproc_per_node=4 dudek/scripts/tdeed.py pretrain   --dataset_path="/workspace/bas/data/broadcast_videos"   --resolution=720 --save_as=pretrained.pt --clip_frames_count=80   --overlap=40

torchrun --nproc_per_node=4 dudek/scripts/tdeed.py train-competition \
    --dataset_path=/workspace/bas/data/competition_videos/ \
    --model_checkpoint_path=/workspace/bas/bt-soccer-bas2/pretrained.pt \
    --resolution=224 \
    --clip_frames_count=80 \
    --overlap=68 \
    --nr_epochs=40 \
    --save_as=tdeed_competition.pt

bas-frame-extract extract-bas-frames \
    --dataset_path=/workspace/bas/data/competition_test/ \
    --resolution=224 \
    --save_all=true \
    --stride=2 \
    --frame_target_width=224 \
    --frame_target_height=224


torchrun --nproc_per_node=4 dudek/scripts/tdeed.py train-competition \
    --dataset_path=/workspace/bas/data/competition_videos/ \
    --model_checkpoint_path=/workspace/bas/bt-soccer-bas2/pretrained.pt \
    --clip_frames_count=160 \
    --overlap=136 \
    --nr_epochs=40 \
    --save_as=tdeed_competition.pt