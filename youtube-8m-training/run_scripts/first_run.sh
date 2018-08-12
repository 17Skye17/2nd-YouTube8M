export CUDA_VISIBLE_DEVICES='0'
python train.py \
--train_data_pattern='hdfs://100.110.18.133:9000/user/VideoAI/rextang/yt8m_2018/frame/*.tfrecord' \
--train_dir='./frame_level_logistic_model' \
--frame_features=True \
--model=FrameLevelLogisticModel \
--feature_names="rgb" \
--feature_sizes="1024" \