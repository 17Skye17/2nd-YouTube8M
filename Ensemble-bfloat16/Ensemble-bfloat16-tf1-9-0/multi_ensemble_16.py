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
from tensorflow.python import pywrap_tensorflow
import tensorflow.contrib.slim as slim
FLAGS = flags.FLAGS

flags.DEFINE_string("Ensemble_Models","","the directory to store models for ensemble.")
flags.DEFINE_string("data_list","./lists/v_dc1.lst","")
flags.DEFINE_string("eval_data_pattern","","")
flags.DEFINE_integer("num_readers",8,"")
flags.DEFINE_integer("batch_size",128,"")
flags.DEFINE_integer("top_k",20,"")
flags.DEFINE_boolean("run_once",True,"")
flags.DEFINE_boolean("restore_once",False,"")
flags.DEFINE_integer("random_seed",0,"")
# TODO
prob_ratio = [1.0,1.0]
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
        files = [l.strip() for l in open(FLAGS.data_list,'r').readlines()]
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
    video_id_batch,model_input_raw,labels_batch,num_frames = get_input_evaluation_tensors(
            reader,   # anyone is ok
            eval_data_pattern,
            batch_size=batch_size,
            num_readers=num_readers
            )
    tf.summary.histogram("model_input_raw",model_input_raw)
    feature_dim = len(model_input_raw.get_shape()) - 1
    model_input = tf.nn.l2_normalize(model_input_raw,feature_dim)
    # two combined predictions
    result = combine_models(models,model_input,num_frames,reader,labels_batch=labels_batch,is_training=False)
    predictions = result["predictions"]
    tf.summary.histogram("model_activations",predictions)

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

def get_latest_checkpoint(filedir):
    index_files = file_io.get_matching_files(os.path.join(filedir,'model.ckpt.bfloat16-*.index'))
    if not index_files:
        return None
    latest_index_file = sorted(
            [(int(os.path.basename(f).split("-")[-1].split(".")[0]),f)
                for f in index_files])[-1][1]
    return latest_index_file[:-6]

def load_vars(sess,train_dir,scope):
    model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scope)
    ckpt = get_latest_checkpoint(os.path.join(FLAGS.Ensemble_Models,train_dir))
    ckpt_reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
    var_to_shape_map = ckpt_reader.get_variable_to_shape_map()
    model_var_size = len(model_vars)
    count = 0
    #for key in var_to_shape_map:
    for v in model_vars:
        for key in var_to_shape_map:
            if len(key.split("tower/")) == 1:
                if v.op.name == key:
                    v.load(ckpt_reader.get_tensor(key),sess)
                    count = count+1
                else:
                    continue
            elif key.split("tower/")[1] == v.op.name.split(scope+"/")[1]:
                    v.load(ckpt_reader.get_tensor(key),sess)
                    print "assign %s to %s" % (key,v.op.name)
                    count = count+1
    if count != model_var_size:
        print "count = %d don't match model var size = %d"%(count,model_var_size)
        import sys
        sys.exit(0)

def evaluation_loop(model_nums,train_dirs,video_id_batch,prediction_batch,label_batch,loss,
                    summary_op,saver,summary_writer,evl_metrics,last_global_step_val):

      global_step_val = -1
      with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
	for i in range(model_nums):
            load_vars(sess,train_dirs[i],"model"+str(i))
        latest_checkpoint = get_latest_checkpoint(os.path.join(FLAGS.Ensemble_Models,train_dirs[0]))  # anyone is OK
        global_step_val = os.path.basename(latest_checkpoint).split("-")[-1]
       # if os.path.exits(save_dir) == False:
       #     os.mkdir(save_dir)
        saver.save(sess,os.path.join(FLAGS.Ensemble_Models+"16_inference","inference_model"))
        if global_step_val == last_global_step_val:
            logging.info("skip this checkpoint global_step_val=%s "
                    "(same as the previous one).",global_step_val)
            return global_step_val

        sess.run([tf.local_variables_initializer()])

        fetches =[video_id_batch,prediction_batch,label_batch,loss,summary_op]
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
                _,predictions_val,labels_val,loss_val, summary_val = sess.run(fetches)
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

                summary_writer.add_summary(summary_val,global_step_val)
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
    tf.set_random_seed(FLAGS.random_seed)
    subdirs = os.listdir(FLAGS.Ensemble_Models)
    subdirs = [subdir for subdir in subdirs if subdir.split(".")[0] != "events"]
    flags_dict = []

    save_flag = True

    bfloat16_dir = []
    for subdir in subdirs:
        subdir = subdir + "/bfloat16"
        bfloat16_dir.append(subdir)
        model_flags_path = os.path.join(os.path.join(FLAGS.Ensemble_Models,subdir),"model_flags.json")
	print "Load model from " + model_flags_path +"\n"
	flags_dict.append(json.loads(file_io.FileIO(model_flags_path,mode="r").read()))
        # save model_flags.json to inference directory
      	if save_flag:
            if os.path.exists(FLAGS.Ensemble_Models+"16_inference") == False:
                os.mkdir(FLAGS.Ensemble_Models+"16_inference")
            shutil.copyfile(model_flags_path,os.path.join(FLAGS.Ensemble_Models+"16_inference","model_flags.json"))
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
        while True:
            last_global_step_val = evaluation_loop(model_nums,bfloat16_dir,video_id_batch,prediction_batch,label_batch,loss,summary_op,saver,summary_writer,evl_metrics,last_global_step_val)
            if FLAGS.run_once:
                break

def main(unused_argv):
    if os.path.exists(os.path.join(FLAGS.Ensemble_Models,"16_inference")):
        shutil.rmtree(os.path.join(FLAGS.Ensemble_Models,"16_inference"))
    utils.set_gpu(1)
    logging.set_verbosity(tf.logging.INFO)
    print("tensorflow version: %s" % tf.__version__)
    #with tf.device("/cpu:0"):
    evaluate()

if __name__ == "__main__":
    app.run()

