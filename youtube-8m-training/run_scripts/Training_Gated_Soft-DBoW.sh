export CUDA_VISIBLE_DEVICES='0'
python ../train.py \
--train_data_pattern='hdfs://100.110.18.133:9000/user/VideoAI/rextang/yt8m_2018/frame/*.tfrecord' \
--video_level_classifier_model=willow_MoeModel \
--model=GatedDbofModelLF \
--train_dir=gateddboflf-4096-1024-80-0002-300iter \
--frame_features=True \
--feature_names="rgb,audio" \
--feature_sizes="1024,128" \
--batch_size=80 \
--base_learning_rate=0.0002 \
--dbof_cluster_size=4096 \
--dbof_hidden_size=1024 \
--moe_l2=1e-6 \
--iterations=300 \
--dbof_relu=False \
--max_step=1000000 \
