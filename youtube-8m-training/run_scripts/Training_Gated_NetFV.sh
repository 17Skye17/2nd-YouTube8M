export CUDA_VISIBLE_DEVICES='1'
python ../train.py \
--train_data_pattern='hdfs://100.110.18.133:9000/user/VideoAI/rextang/yt8m_2018/frame/*.tfrecord' \
--video_level_classifier_model=willow_MoeModel \
--model=NetFVModelLF \
--train_dir=gatednetfvLF-128k-1024-80-0002-300iter-norelu-basic-gatedmoe \
--frame_features=True \
--feature_names="rgb,audio" \
--feature_sizes="1024,128" \
--batch_size=80 \
--base_learning_rate=0.0002 \
--fv_cluster_size=128 \
--fv_hidden_size=1024 \
--moe_l2=1e-6 \
--iterations=300 \
--learning_rate_decay=0.8 \
--fv_relu=False \
--gating=True \
--moe_prob_gating=True \
--fv_couple_weights=False \
--max_step=600000 \
