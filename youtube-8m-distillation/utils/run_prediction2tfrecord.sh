export CUDA_VISIBLE_DEVICES=''
python p2tf.py \
--data_dir='/mnt/ceph_cv/aicv_image_data/forestlma/Youtube8M_2018/frame' \
--output_path='./tmp' \
--input_file_pattern='validate0000.tfrecord' \
--csv_file='' \
--number_of_classes=3862 \
--number_of_videos='1013313' \