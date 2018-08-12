import tensorflow as tf
import utils
import shutil
import losses
import glob
import json
import pickle
import os
import time
import losses
import numpy as np
import readers
import frame_level_models
import video_level_models
import eval_util
from tensorflow.python.lib.io import file_io
from tensorflow import app
from tensorflow import logging
from tensorflow import flags
from tensorflow import gfile
from datetime import datetime
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import device_lib
FLAGS = flags.FLAGS

flags.DEFINE_string("Ensemble_Models","./","the directory to store models for ensemble.")
flags.DEFINE_string("model_path",None,"the files to store models for ensemble.")
flags.DEFINE_string("output_path",None,"the files to store ensembled models.")
flags.DEFINE_string("eval_data_pattern","","")
flags.DEFINE_integer("num_readers",8,"")
flags.DEFINE_integer("batch_size",128,"")
flags.DEFINE_integer("top_k",20,"")
flags.DEFINE_boolean("run_once",True,"")
flags.DEFINE_boolean("restore_once",False,"")
flags.DEFINE_integer("random_seed",666,"")
tf.set_random_seed(FLAGS.random_seed)
# TODO
prob_ratio = [1.0,1.0,1.0,1.0,1.0,1.0]

def find_class_by_name(name,modules):
    modules = [getattr(module,name,None) for module in modules]
    return next(a for a in modules if a)

def combine_models(models,model_input,num_frames,reader,labels_batch,is_training):
    model_nums = len(models)
    predictions = 0
    for i in range(model_nums):
        with tf.variable_scope("model"+str(i)):
            #TODO
            #FLAGS.name = name[i]
            #print flags
            model_prob = models[i].create_model(model_input=model_input,
                               num_frames=num_frames,
                               vocab_size=reader.num_classes,
                               labels=labels_batch,
                               is_training=is_training)
            predictions = predictions + prob_ratio[i]*model_prob["predictions"]
            # destory flags
    return {"predictions":predictions}

def get_input_evaluation_tensors(reader,
                                data_pattern,
                                batch_size=1024,
                                num_readers=8
                                ):
    logging.info("Using batch size of " + str(batch_size) + " for evaluation.")
    with tf.name_scope("eval_input"):
        files = gfile.Glob(data_pattern)
        if not files:
            raise IOError("Unable to find the evaluation files.")
        logging.info("number of evaluation files: " + str(len(files)))

        filename_queue = tf.train.string_input_producer(files,shuffle=False,num_epochs=1)
        eval_data = [
                reader.prepare_reader(filename_queue) for _ in range(num_readers)
                ]
        return tf.train.batch_join(
                eval_data,
                batch_size=batch_size,
                capacity=5 * batch_size,
                allow_smaller_final_batch=True,
                enqueue_many=True
                )


def  build_graph(reader,
                 models,
                 eval_data_pattern,
                 label_loss_fns,
                 batch_size=1024,
                 num_readers=1):

    global_step = tf.Variable(0,trainable=False,name="global_step")
    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    gpus = gpus[:FLAGS.num_gpu]
    num_gpus = len(gpus)

    if num_gpus > 0:
        logging.info("Using the following GPUs to train: " + str(gpus))
        num_towers = num_gpus
        device_string = '/gpu:%d'
    else:
        logging.info("No GPUs found. Training on CPU.")
        num_towers = 1
        device_string = '/cpu:%d'


    video_id_batch,model_input_raw,labels_batch,num_frames = get_input_evaluation_tensors(
            reader,   # anyone is ok
            eval_data_pattern,
            batch_size=batch_size,
            num_readers=num_readers
            )
    tf.summary.histogram("model_input_raw",model_input_raw)
    feature_dim = len(model_input_raw.get_shape()) - 1
    model_input = tf.nn.l2_normalize(model_input_raw,feature_dim)


    tower_inputs = tf.split(model_input, num_towers)
    tower_labels = tf.split(labels_batch, num_towers)
    tower_num_frames = tf.split(num_frames, num_towers)
    tower_predictions = []
    tower_label_losses = []
    # two combined predictions
    for i in range(num_towers):
        with tf.device(device_string % i):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            result = combine_models(models,model_input,num_frames,reader,labels_batch=labels_batch,is_training=False)
            predictions = result["predictions"]
            # tf.summary.histogram("model_activations",predictions)

            if "loss" in result.keys():
               label_loss = result["loss"]
            else:
               label_loss = label_loss_fns[0].calculate_loss(predictions,labels_batch)

    tf.add_to_collection("global_step",global_step)
    tf.add_to_collection("loss",label_loss)
    tf.add_to_collection("predictions",predictions)
    tf.add_to_collection("input_batch",model_input)
    tf.add_to_collection("input_batch_raw",model_input_raw)
    tf.add_to_collection("video_id_batch",video_id_batch)
    tf.add_to_collection("num_frames",num_frames)
    tf.add_to_collection("labels",tf.cast(labels_batch,tf.float32))
    tf.add_to_collection("summary_op",tf.summary.merge_all())



def get_params(flags_dict):

    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
            flags_dict["feature_names"],flags_dict["feature_sizes"]
            )
    if flags_dict["frame_features"]:
        reader = readers.YT8MFrameFeatureReader(feature_names=feature_names,
                                                feature_sizes=feature_sizes)
    else:

        reader = readers.YT8MAggregatedFeatureReader(feature_names=feature_names,
                                                feature_sizes=feature_sizes)
    model  = find_class_by_name(flags_dict["model"],[frame_level_models,video_level_models])()

    # a model instance is created
    label_loss_fn = find_class_by_name(flags_dict["label_loss"],[losses])()
    return model,label_loss_fn,reader

# def get_latest_checkpoint(filedir):
#     index_files = file_io.get_matching_files(os.path.join(filedir,'model.ckpt-*.index'))
#     if not index_files:
#         return None
#     latest_index_file = sorted(
#             [(int(os.path.basename(f).split("-")[-1].split(".")[0]),f)
#                 for f in index_files])[-1][1]
#     return latest_index_file[:-6]

def get_latest_checkpoint(filedir):
    index_files = file_io.get_matching_files(os.path.join(filedir,'model.ckpt-*.index'))
    if not index_files:
        return None
    latest_index_file = sorted(
            [(int(os.path.basename(f).split("-")[-1].split(".")[0]),f)
                for f in index_files])[-1][1]
    return latest_index_file[:-6]

def load_vars(sess,train_dir,scope):
    model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scope)
    model_var_size = len(model_vars)
    print('scope: %s, number of global_variables: %d' % (scope, model_var_size))
    # ckpt = get_latest_checkpoint(os.path.join(FLAGS.Ensemble_Models,train_dir))

    reader = pywrap_tensorflow.NewCheckpointReader(os.path.join(FLAGS.Ensemble_Models,train_dir,'model.ckpt.bfloat16-1'))
    #reader = pywrap_tensorflow.NewCheckpointReader(os.path.join(FLAGS.Ensemble_Models,train_dir,'model.ckpt-1'))
    keys = reader.get_variable_to_shape_map()
    print('checkpoint_file: %s, number of variables in checkpoint_file: %d' % (train_dir, len(keys)))

    count=0
    for key in keys:
        var_val = reader.get_tensor(key)
        for k in model_vars:
            if len(key.split("tower/")) == 1:
                if k.op.name == key:
                    k.load(var_val,sess)
                    count = count+1
                else:
                     continue
            elif key.split("tower/")[1] == k.op.name.split(scope+"/")[1]:
#               # sess.run(k.assign(var_val))  assign will store variables to meta graph!!!!!
                    k.load(var_val,sess)
                    print "assign %s to %s" % (key,k.op.name)
                    count = count+1
    if count != model_var_size:
        print "count = %d  don't match model var size= %d"%(count,model_var_size)
        import sys
        sys.exit(0)

# def load_vars(sess,train_dir,scope):
#     model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scope)
#     ckpt = get_latest_checkpoint(os.path.join(FLAGS.Ensemble_Models,train_dir))
#     f = open(ckpt+".npy",'r')
#     ckpt_vars = pickle.load(f)
#     keys = ckpt_vars.keys()
# #    for key in keys:
# #        print "name = %s   shape= %s"%(key,ckpt_vars[key].shape)
# #    print "\n\n\n"
# #    for k in model_vars:
# #
# #        print "name = %s   shape= %s"%(k.op.name,k.shape)
# #    import sys
# #    sys.exit(0)
#     model_var_size = len(model_vars)
#     count=0
#     for key in keys:
#         var_val = ckpt_vars[key]
#         for k in model_vars:
#             if len(key.split("tower/")) == 1:
#                 if k.op.name == key:
#                     k.load(var_val,sess)
#                     count = count+1
#                 else:
#                      continue
#             elif key.split("tower/")[1] == k.op.name.split(scope+"/")[1]:
# #               # sess.run(k.assign(var_val))  assign will store variables to meta graph!!!!!
#                     k.load(var_val,sess)
#                     print "assign %s to %s" % (key,k.op.name)
#                     count = count+1
#     if count != model_var_size:
#         print "count = %d  don't match model var size= %d"%(count,model_var_size)
#         import sys
#         sys.exit(0)

def evaluation_loop(model_nums,train_dirs,video_id_batch,prediction_batch,label_batch,loss,
                    summary_op,saver,summary_writer,evl_metrics):

    global_step_val = -1
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        for i in range(model_nums):
            load_vars(sess,train_dirs[i],"model"+str(i))
        # new load
        saver.save(sess,os.path.join(FLAGS.Ensemble_Models+FLAGS.output_path,"inference_model"))


        sess.run([tf.local_variables_initializer()])

        fetches =[video_id_batch,prediction_batch,label_batch,loss]
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(
                sess,coord=coord,daemon=True,start=True
                ))
            logging.info("enter eval_once loop global_step_val = %s.",
                        global_step_val)
            evl_metrics.clear()

            examples_processed = 0
            while not coord.should_stop():
                batch_start_time = time.time()
                _,predictions_val,labels_val,loss_val = sess.run(fetches)
                seconds_per_batch = time.time() - batch_start_time
                example_per_second = labels_val.shape[0] / seconds_per_batch
                examples_processed += labels_val.shape[0]

                iteration_info_dict = evl_metrics.accumulate(predictions_val,labels_val,loss_val)
                iteration_info_dict["examples_per_second"] = example_per_second

                iterinfo = utils.AddGlobalStepSummary(
                        summary_writer,
                        global_step_val,
                        iteration_info_dict,
                        summary_scope="Eval"
                        )
                logging.info("examples_processed: %d | %s",examples_processed,iterinfo)
        except tf.errors.OutOfRangeError as e:
            logging.info(
                    "Done with batched inference. Now calculating global performance metrics."
                    )
            epoch_info_dict = evl_metrics.get()
            epoch_info_dict["epoch_id"] = global_step_val

            #summary_writer.add_summary(summary_val,global_step_val)
            epochinfo = utils.AddEpochSummary(
                    summary_writer,
                    global_step_val,
                    epoch_info_dict,
                    summary_scope="Eval"
                    )
            logging.info(epochinfo)
            evl_metrics.clear()
        except Exception as e:
            logging.info("Unexpected exception:" +str(e))
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads,stop_grace_period_secs=10)

        return global_step_val


def evaluate():
    if os.path.exists(FLAGS.Ensemble_Models+FLAGS.output_path):
        shutil.rmtree(FLAGS.Ensemble_Models+FLAGS.output_path)
    # subdirs = [x for x in os.listdir(FLAGS.Ensemble_Models) if os.path.isdir(x)]
    subdirs = [os.path.join(FLAGS.Ensemble_Models, x) for x in FLAGS.model_path.split(',')]
    flags_dict = []
    save_flag = True
    for subdir in subdirs:
        model_flags_path = os.path.join(os.path.join(FLAGS.Ensemble_Models,subdir),"model_flags.json")
        print "Load model from " + model_flags_path +"\n"
        flags_dict.append(json.loads(file_io.FileIO(model_flags_path,mode="r").read()))
        # save model_flags.json to inference dictory
        if save_flag:
            if os.path.exists(FLAGS.Ensemble_Models+FLAGS.output_path) == False:
                os.mkdir(FLAGS.Ensemble_Models+FLAGS.output_path)
            shutil.copyfile(model_flags_path,os.path.join(FLAGS.Ensemble_Models+FLAGS.output_path,"model_flags.json"))
            save_flag = False

    g = tf.Graph()
    with g.as_default():
        models = []
        label_loss_fns = []
        readers = []
        model_nums = len(subdirs)
        for m in range(model_nums):
            model,label_loss_fn,reader = get_params(flags_dict[m])
            models.append(model)
            label_loss_fns.append(label_loss_fn)
            readers.append(reader)
        #model2,label_loss_fn2,reader2 = get_params(flags_dict2)
        # start build graph
        build_graph(reader=readers[0], # anyone is ok
                    models=models,
                    eval_data_pattern=FLAGS.eval_data_pattern,
                    label_loss_fns=label_loss_fns,
                    batch_size=FLAGS.batch_size,
                    num_readers=FLAGS.num_readers)

        logging.info("built evaluation graph")
        video_id_batch = tf.get_collection("video_id_batch")[0]
        prediction_batch = tf.get_collection("predictions")[0]
        label_batch = tf.get_collection("labels")[0]
        loss = tf.get_collection("loss")[0]
        summary_op = tf.get_collection("summary_op")[0]

        all_vars = tf.global_variables()
        # remove global_step or inference can't find uninitialize parameter
        all_vars = [v for v in all_vars if "global_step" not in v.op.name]
        saver = tf.train.Saver(all_vars)

        evl_metrics = eval_util.EvaluationMetrics(readers[0].num_classes,FLAGS.top_k)
        summary_writer = tf.summary.FileWriter(
                FLAGS.Ensemble_Models,graph=tf.get_default_graph())

        last_global_step_val = -1
        last_global_step_val = evaluation_loop(model_nums,subdirs,video_id_batch,prediction_batch,label_batch,loss,summary_op,saver,summary_writer,evl_metrics)


def main(unused_argv):
    # utils.set_gpu(1)
    logging.set_verbosity(tf.logging.INFO)
    print("tensorflow version: %s" % tf.__version__)
    evaluate()

if __name__ == "__main__":
    app.run()

