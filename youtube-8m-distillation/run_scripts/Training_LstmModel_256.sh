export CUDA_VISIBLE_DEVICES='3'
python ../train.py \
--train_data_pattern='hdfs://100.110.18.133:9000/user/VideoAI/rextang/yt8m_2018/frame/*.tfrecord' \
--model=LstmModel \
--train_dir=lstm-0002-val-150-random-256 \
--frame_features=True \
--feature_names="rgb,audio" \
--feature_sizes="1024,128" \
--batch_size=128 \
--base_learning_rate=0.0002 \
--iterations=150 \
--lstm_random_sequence=True \
--max_step=400000 \
--lstm_cells=256 \
--num_epochs=20 \
--moe_num_mixtures=4 \