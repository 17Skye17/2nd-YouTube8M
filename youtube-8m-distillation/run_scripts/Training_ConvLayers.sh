export CUDA_VISIBLE_DEVICES='0'
export AUTH_URL="http://10.242.152.100:8008/checkAuthorityForSZHadoop.php"
export AUTH_HDFS_URL="http://10.242.152.100:8008/checkHdfsPathAuthority.php"
export RTX_NAME=matthzhuang
export BUS_NAME=VideoAI
export TOKEN=e7412d14-f0e8-43a2-a84c-23fa418d5399
export HADOOP_HDFS_HOME='./hadoop-2.5.1'
export HADOOP_HOME=$HADOOP_HDFS_HOME
export HADOOP_CONF_DIR="$HADOOP_HDFS_HOME/etc/hadoop"
export HADOOP_CLASSPATH="`$HADOOP_HOME/bin/hadoop classpath`"
export CLASSPATH="$HADOOP_CLASSPATH:$CLASSPATH"
for i in `find ${HADOOP_HOME} -name "*.jar"`
do
       export CLASSPATH="$i:$CLASSPATH"
done
export http_proxy=http://10.223.133.20:52107
export https_proxy=http://10.223.133.20:52107

python ../train.py \
--train_data_pattern='hdfs://100.110.18.133:9000/user/VideoAI/rextang/yt8m_2018/frame/*.tfrecord' \
--model=ConvSquare \
--video_level_classifier_model=willow_MoeModel \
--train_dir=convsquare-convlayer-1-0002-basic-gatedmoe \
--frame_features=True \
--feature_names="rgb,audio" \
--feature_sizes="1024,128" \
--batch_size=80 \
--base_learning_rate=0.0002 \
--moe_l2=1e-6 \
--iterations=300 \
--learning_rate_decay=0.8 \
--ConvLayers=1 \
--Conv_temporal_field=2 \
--gating=True \
--moe_prob_gating=True \
--graphvlad=True \
--max_step=700000 \