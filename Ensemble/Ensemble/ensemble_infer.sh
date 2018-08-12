export CUDA_VISIBLE_DEVICES="0"
python inference.py \
    --train_dir='inference_models' \
    --input_data_pattern='/DATACENTER/1/forwchen/YT8M/test/test*.tfrecord' \
    --output_file="samples_submission.csv" \
