export CUDA_VISIBLE_DEVICES='0'
python ../train.py \
    --train_data_pattern='/DATACENTER/1/forwchen/YT8M/train/train*.tfrecord' \
    --model=SoftDbofModelLF_8k \
    --train_dir='/home/skye/youtube_log/softdboflf-8000-1024-80-0002-300iter' \
    --frame_features=True \
    --start_new_model=True \
    --feature_names="rgb,audio" \
    --feature_sizes="1024,128" \
    --batch_size=80 \
    --base_learning_rate=0.0002 \
    --iterations=300 \
    --max_step=800000 \
