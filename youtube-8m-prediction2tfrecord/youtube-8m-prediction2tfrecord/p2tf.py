import sys
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
import time
import numpy as np
import pandas

tf.flags.DEFINE_string("data_dir", '/mnt/ceph_cv/aicv_image_data/forestlma/Youtube8M_2018/frame',
                       "Directory to read tfrecoreds from")

tf.flags.DEFINE_string("output_path", './tmp',
                       "Directory to read tfrecoreds from")
tf.flags.DEFINE_string("input_file_pattern", 'validate0000.tfrecord',
                       "Dinput_file_pattern")
tf.flags.DEFINE_string("csv_file", None,
                       "Directory to read csv_file")
tf.flags.DEFINE_integer("number_of_classes", 3862,
                        "number_of_classes")
tf.flags.DEFINE_integer("number_of_videos", 1013313,
                        "number_of_videos")
tf.flags.DEFINE_integer("bstart", 0,
                        "number_of_classes")
tf.flags.DEFINE_integer("bend", 0,
                        "number_of_classes")
# test 1133323 eval0-3500 1013313 train 3888919
FLAGS = tf.flags.FLAGS


def main():
    tfrecords_path = gfile.Glob(os.path.join(FLAGS.data_dir, FLAGS.input_file_pattern))
    tfrecords_path.sort()
    tfrecords_path = tfrecords_path[FLAGS.bstart:FLAGS.bend]
    print('Processing %d-%d' %(FLAGS.bstart,FLAGS.bend))
    # tfrecords_path = os.path.join(FLAGS.data_dir, FLAGS.input_file_pattern)
    file_temp_path = FLAGS.output_path
    if not os.path.isdir(file_temp_path):
        os.makedirs(file_temp_path)

    csv_content = pandas.read_csv(FLAGS.csv_file)
    num_videos = len(csv_content['VideoId'])
    count = 0
    for file_ in tfrecords_path:
        video_name_str = 'distill_'+file_.split('/')[-1]
        tfrecord_name = os.path.join(file_temp_path, video_name_str)
        writer = tf.python_io.TFRecordWriter(tfrecord_name)
        record_iterator = tf.python_io.tf_record_iterator(path=file_)
        print('Processing: %s' % file_.split('/')[-1])
        for string_record in record_iterator:
            example = tf.train.SequenceExample()
            example.ParseFromString(string_record)
            feature = {}
            feature['rgb'] = tf.train.FeatureList(feature=[x for x in example.feature_lists.feature_list['rgb'].feature])
            feature['audio'] = tf.train.FeatureList(feature=[x for x in example.feature_lists.feature_list['audio'].feature])

            context = {}
            context['id'] = tf.train.Feature(bytes_list=example.context.feature['id'].bytes_list)
            context['labels'] = tf.train.Feature(int64_list=example.context.feature['labels'].int64_list)

            video_id = example.context.feature['id'].bytes_list.value[0].decode("utf-8")
            csv_object = csv_content.loc[csv_content['VideoId'] == video_id]
            prediction_pair = csv_object['LabelConfidencePairs'].values[0].split(' ')
            encoded_predictions = np.zeros(FLAGS.number_of_classes, dtype=np.float32)
            for prediction_index in range(0, len(prediction_pair) // 2, 2):
                label_index, label_score = prediction_pair[prediction_index], prediction_pair[prediction_index + 1]
                encoded_predictions[int(label_index)] = float(label_score)
            context['ensemble_labels'] = _bytes_feature(encoded_predictions.tostring())
            #context['ensemble_labels'] = tf.train.Feature(float_list=tf.train.FloatList(value=encoded_predictions))

            example_out = tf.train.SequenceExample(context=tf.train.Features(feature=context),
                                                   feature_lists=tf.train.FeatureLists(feature_list=feature))
            writer.write(example_out.SerializeToString())
        writer.close()
        count += 1


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def parse_example_proto(example_serialized):
    contexts, features = tf.parse_single_sequence_example(
        example_serialized,
        context_features={"id": tf.FixedLenFeature(
            [], tf.string),
            "labels": tf.VarLenFeature(tf.int64)},
        sequence_features={
            feature_name: tf.FixedLenSequenceFeature([], dtype=tf.string)
            for feature_name in ['rgb', 'audio']
        })

    return contexts, features


# load model
if __name__ == "__main__":
    main()
