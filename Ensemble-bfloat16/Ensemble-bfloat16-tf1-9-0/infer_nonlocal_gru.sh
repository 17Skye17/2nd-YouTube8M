python inference.py \
    --train_dir=$1 \
    --output_file='test_multi_ensemble/sample_submission.csv' \
    --output_model_tgz='test_multi_ensemble/output.tgz' \
    --input_data_pattern=/DATACENTER/1/forwchen/YT8M/test/test*.tfrecord \
    --batch_size=128 \
    --gru_cells=1200

