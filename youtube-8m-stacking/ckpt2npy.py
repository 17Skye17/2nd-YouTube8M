import tensorflow as tf
from tensorflow import flags
from tensorflow.python.lib.io import file_io
from tensorflow import app
import pickle
import os

FLAGS=flags.FLAGS
flags.DEFINE_string("model_dir","../youtube_log/frame_level_gru_07051606","the directory to store model ckpts")

def get_latest_checkpoint(filedir):
    index_files = file_io.get_matching_files(os.path.join(filedir,'model.ckpt-*.index'))
    if not index_files:
        return None
    latest_index_file = sorted(
            [(int(os.path.basename(f).split("-")[-1].split(".")[0]),f)
                for f in index_files])[-1][1]
    return latest_index_file[:-6]

def main(unused_argv):

    with tf.Session() as sess:
    #    ckpt = get_latest_checkpoint(FLAGS.model_dir)
        ckpt = os.path.join(FLAGS.model_dir,"inference_model")
        meta = tf.train.import_meta_graph(ckpt+".meta",clear_devices=True)
        ckpt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        saver = tf.train.Saver(var_list=ckpt_vars)
        saver.restore(sess,ckpt)
        print "\n\nvars in ckpt file...\n"
        for v in  ckpt_vars:
            print v.op.name
        _dict = {}
        for v in ckpt_vars:
            _dict[v.op.name] = v.eval(sess)
        print "\n\nvars in saved dict, check save...\n"
        for key in _dict.keys():
            print key
    save_file = ckpt+".npy"
    f = open(save_file,"w")
    pickle.dump(_dict,f,0) # write as txt


if __name__ == "__main__":
    app.run()
