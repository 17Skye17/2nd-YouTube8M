export CUDA_VISIBLE_DEVICES='3'
python ../train.py \
--train_data_pattern='/data1/rextang/yt8m/frame/*.tfrecord' \
--model=LstmModel \
--train_dir=lstm-0002-val-150-random \
--frame_features=True \
--feature_names="rgb,audio" \
--feature_sizes="1024,128" \
--batch_size=128 \
--base_learning_rate=0.0002 \
--iterations=150 \
--lstm_random_sequence=True \
--max_step=400000 \