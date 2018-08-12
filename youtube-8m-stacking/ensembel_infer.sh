export CUDA_VISIBLE_DEVICES='1'
python inference.py \
--train_dir='nonlocalVLAD_GRU_NetFV_BoW' \
--input_data_pattern='hdfs://100.110.18.133:9000/user/VideoAI/rextang/yt8m_2018/frame/test/*.tfrecord' \
--output_file='nonlocalVLAD_GRU_NetFV_BoW/nonlocalVLAD_GRU_NetFV_BoW.csv' \