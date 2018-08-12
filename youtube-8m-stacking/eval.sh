export CUDA_VISIBLE_DEVICES='1'
python multi_ensemble.py \
--model_path='embedgauNonLocal-gatednetvladLF-64k-1024-80-0002-300iter-norelu-basic-gatedmoe8-t64,GRU-0002-1200-backup,gatednetfvLF-32k-1024-80-0002-300iter-norelu-basic-gatedmoe4_test,gateddboflf-2048-1024-80-0002-300iter_test' \
--output_path='nonlocalVLAD_GRU_NetFV_BoW' \
--eval_data_pattern='hdfs://100.110.18.133:9000/user/VideoAI/rextang/yt8m_2018/frame/val/*.tfrecord' \