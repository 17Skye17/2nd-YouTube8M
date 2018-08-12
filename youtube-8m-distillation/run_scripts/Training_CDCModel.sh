export CUDA_VISIBLE_DEVICES='1'
python ../train.py \
--train_data_pattern='/data1/rextang/yt8m/frame/*.tfrecord' \
--model=CDCModel \
--train_dir=CDC-0002-1200 \
--frame_features=True \
--feature_names="rgb,audio" \
--feature_sizes="1024,128" \
--batch_size=128 \
--base_learning_rate=0.0002 \
--gru_cells=1200 \
--learning_rate_decay=0.9 \
--moe_l2=1e-6 \
--max_step=300000 \
--gru_random_sequence=True \
--iterations=64 \