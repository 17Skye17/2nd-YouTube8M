export CUDA_VISIBLE_DEVICES="0"
python ../train.py \
    --train_data_pattern='/DATACENTER/1/forwchen/YT8M/train/train*.tfrecord' \
    --model=early_NetVLADModelLF \
    --train_dir='/home/skye/youtube_log/nonlocal-earlyNetVLADModel' \
    --frame_features=True \
    --feature_names="rgb,audio" \
    --feature_sizes="1024,128" \
    --batch_size=80 \
    --base_learning_rate=0.0002 \
    --learning_rate_decay=0.8 \
    --max_step=700000 \
