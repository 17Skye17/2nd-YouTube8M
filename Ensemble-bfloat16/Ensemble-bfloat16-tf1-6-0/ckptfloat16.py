from tensorflow.python import pywrap_tensorflow
import os
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow import flags
import shutil
FLAGS=flags.FLAGS

flags.DEFINE_string("model_dir","","")

def get_latest_checkpoint(filedir):
    index_files = file_io.get_matching_files(os.path.join(filedir,"model.ckpt-*.index"))
    if not index_files:
        return None
    latest_index_file = sorted(
            [(int(os.path.basename(f).split("-")[-1].split(".")[0]),f)
                for f in index_files])[-1][1]
    return latest_index_file[:-6]

if __name__ == "__main__":
    bfloat16_dir = os.path.join(FLAGS.model_dir,"bfloat16")
    if os.path.exists(bfloat16_dir) == False:
        os.mkdir(bfloat16_dir)
    shutil.copyfile(os.path.join(FLAGS.model_dir,"model_flags.json"),os.path.join(bfloat16_dir,"model_flags.json"))
    g = tf.Graph()
    with g.as_default():
      with tf.Session() as sess:
        checkpoint_path = get_latest_checkpoint(FLAGS.model_dir)
        meta = tf.train.import_meta_graph(checkpoint_path+".meta",clear_devices=True)
        saver = tf.train.Saver()
        saver.restore(sess,checkpoint_path)
        saver1 = tf.train.Saver(tf.global_variables())
        saver1.save(sess,os.path.join(bfloat16_dir,"model.ckpt.bfloat16-1"))
