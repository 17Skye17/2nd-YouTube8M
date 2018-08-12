1.train a single model from scratch

cd  youtube-8m-training/run_scripts
./nonlocal-NetVLADModel.sh


2.linear checkpoint averaging

./SWA_utils/avg_eval.sh  [a folder containing single model's checkpoints]


3.convert tf.float32 models to tf.bfloat16

cd Ensemble/Ensemble/
./ckptfloat16.sh  [a folder containing single model's checkpoints]


4.ensemble all models

cd Ensemble/Ensemble/
./multi_ensemble_tile.sh [a folder containing model-bfloat16 folders]



5.inference

./ensemble_infer.sh
