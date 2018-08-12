export CUDA_VISIBLE_DEVICES=''
python p2tf.py \
--data_dir='/mnt/ceph_cv/aicv_image_data/forestlma/Youtube8M_2018/frame' \
--output_path='/mnt/ceph_cv/aicv_image_data/forestlma/Youtube8M_2018/distillation_frame' \
--input_file_pattern='train*.tfrecord' \
--csv_file='nonlocalVLAD_AVG-GRU_AVG-FV-Bow-LSTM-bfloat16_seed2018_train.csv' \
--number_of_classes=3862 \
--number_of_videos='3888919' \
--bstart=3000 \
--bend=3500 \
