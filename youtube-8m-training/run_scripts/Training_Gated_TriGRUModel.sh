export CUDA_VISIBLE_DEVICES='1'
python ../train.py \
--train_data_pattern='/mnt/ceph_cv/aicv_image_data/forestlma/Youtube8M_2018/frame/*.tfrecord' \
--model=TriGruModel \
--train_dir=TriGRU-0002-1200-moe4 \
--video_level_classifier_model=willow_MoeModel \
--frame_features=True \
--feature_names="rgb,audio" \
--feature_sizes="1024,128" \
--batch_size=128 \
--base_learning_rate=0.0002 \
--gru_cells=1200 \
--learning_rate_decay=0.9 \
--moe_l2=1e-6 \
--max_step=700000 \
--moe_prob_gating=True \
--num_epochs=20 \
--moe_num_mixtures=4 \