export CUDA_VISIBLE_DEVICES="0"
python multi_ensemble_tile.py \
    --Ensemble_Models=$1 \
    --batch_size=80 \
    --model_path='embedgauNonLocal-gatednetvladLF-64k-1024-80-0002-300iter-norelu-basic-gatedmoe8-t64-300k-bfloat16,GRU-002-1200-backup-bfloat16' \
    --output_path='nonlocalVLAD_GRU' \
    --eval_data_pattern='/DATACENTER/1/forwchen/YT8M/val/validate*.tfrecord' \
