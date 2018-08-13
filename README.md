# YT8M-T solution for <2nd YouTube-8M Video Understanding Challenge> competition

Below you can find a outline of how to reproduce our solution for the <2nd YouTube-8M Video Understanding Challenge> competition.
If you run into any trouble with the setup/code or have any questions please contact us at <303297957@qq.com>

****
# CONTENTS
* [Archive Contens](#archive-contents)
* [Hardware](#hardware)
* [Software](#software)
* [Installation](#installation)
* [Training](#training)
	* [Data Setup](#data-setup)
	* [Train a Single Model](#train-a-single-model)
	* [Linear Checkpoint Averaging](#linear-checkpoint-averaging)
* [Evaluation](#evaluation)
	* [Quantization](#quantization)
	* [Model Ensemble](#model-ensemble)
* [Inference](#inference)
* [License](#license)

# ARCHIVE CONTENTS
| Folder | Contents|
| ---------- | -----------|
| Ensemble  |  model ensemble code for final submission   |
| Ensemble-bfloat16   | modified tensorflow source code which contains variable_scope.py and saver.py, for load and save float32 models to bfloat16 models   |
|SWA_utils|code to do linear checkpoint averaging among several checkpoints|
|youtube-8m-distillation|code to train ensembled models with distillation method|
|youtube-8m-prediction2tfrecord|code to write predicted labels to tfrecord for distillation|
|youtube-8m-stacking|code to ensemble models via stacking|
|youtube-8m-training|code to train a single model|

# HARDWARE
|Hardware   |Specs   |
| ------------ | ------------ |
|  OS | Ubuntu 16.04 LTS (256 GB boot disk) |
|GPU|4x NVIDIA GeForce GTX 1080 Ti|
|CPU|2x Intel(R) Xeon(R) CPU E5-2630 v4 (20 cores in total)|
|  memory |  128G |

# SOFTWARE 
| Software  | Version  |
| ------------ | ------------ |
| Python  | 2.7.15  |
| CUDA  | 9.0  |
|cudnn|7.0.5|
|nvidia derivers|v.384|
|tensorflow|1.9.0|

# INSTALLATION
### 1.install tensorflow-1.9.0
we recommand installing tensorflow 1.9.0 which has better support for tf.bfloat16 type.
Please follow installation guide on [tensorflow website](https://www.tensorflow.org/install/install_linux? "tensorflow website").
### 2.install requirements
To install some python packages, simply run:
```shell
pip install requirements.txt
```
# TRAINING
We finally used 6 models, they are listed below:

| Model  | Train Scripts  |
| ------------ | ------------ |
| GRUModel  | GRUModel.sh  |
|  nonlocal-earlyNetVLADModel |  nonlocal-earlyNetVLADModel.sh |
|  nonlocal-LightNetVLADModel |  nonlocal-LightNetVLADModel.sh |
|  nonlocal-NetVLADModel |  nonlocal-NetVLADModel.sh |
|  Soft-DBoF-4k | Soft-DBoF-4k.sh  |
|  Soft-DBoF-8k | Soft-DBoF-8k.sh  |

### DATA SETUP
Please follow [youtube-8m github](https://github.com/google/youtube-8m "youtube-8m github") to download dataset.
### TRAIN A SINGLE MODEL
To train a single model from scrach, for example:
```shell
cd  youtube-8m-training/run_scripts
./nonlocal-NetVLADModel.sh
```
Note:You may find many run scripts, but currently, we only provide 6 useful models listed above.

### LINEAR CHECKPOINT AVERAGING
Our models got boosted by doing linear checkpoint averaging among several checkpoints.
First, modify `SWA_utils/avg_eval.sh` according to different models with `swa=True` and adjust `swa_start`.
Then run:
```shell
./SWA_utils/avg_eval.sh  [a folder containing single model's checkpoints]
```

# EVALUATION
During evaluation, we are going to ensemble these single models by rebuilding graph and  adding prediction results.

### QUANTIZATION
To downsize these models, we convert the pretrained tf.float32 models to tf.bfloat16 by modifying code in `variable_scope.py` and `saver.py`.(Located in `/usr/lib/python2.7/site-packages/tensorflow/python/ops` and `/usr/lib/python2.7/site-packages/tensorflow/python/training`)

For tensorflow-1.9.0, please replace `variable_scope.py` and `saver.py` with `Youtube8M-Code-Final/Ensemble-bfloat16/Ensemble-bfloat16-tf1-9-0/variable_scope.py` and `Youtube8M-Code-Final/Ensemble-bfloat16/Ensemble-bfloat16-tf1-9-0/saver.py`

For tensorflow-1.6.0, please replace `variable_scope.py` and `saver.py` with `Youtube8M-Code-Final/Ensemble-bfloat16/Ensemble-bfloat16-tf1-6-0/variable_scope.py` and `Youtube8M-Code-Final/Ensemble-bfloat16/Ensemble-bfloat16-tf1-6-0/saver.py`

### MODEL ENSEMBLE
After replacing `variable_scope.py` and `saver.py`, now we can ensemble these models.
```shell
cd Ensemble/Ensemble/
./ckptfloat16.sh  [a folder containing single model's checkpoints]
```
Now, we generated a `bfloat16` folder which contains tf.bfloat16 checkpoints, the directory structure is:
./allmodels/model1/bfloat16
./allmodels/model2/bfloat16
......

**Put all bfloat16 floders in a new floder**, then modify `model_path` and `output_path` in `multi_ensemble_tile.sh`.
**Now, we are ready to ensemble these models**. **Please run:**
```shell
cd Ensemble/Ensemble/
./multi_ensemble_tile.sh [a folder containing model-bfloat16 folders]
```
Done! A final inference model is generated in `output_path`

# INFERENCE
First modify `train_dir` and `output_file` in `ensemble_infer.sh`,then run:
```shell
./ensemble_infer.sh
```
You can also use our pretrained model to make predictions, the pretrained` inferece_model` is in  `./models`, just specify `train_dir=./models`.

Done! Now we get the .csv file for submission : )

# LICENSE
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
