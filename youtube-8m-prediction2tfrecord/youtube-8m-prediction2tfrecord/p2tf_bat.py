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
# test 1133323 eval0-3500 1013313 train 3888919
FLAGS = tf.flags.FLAGS


def main():
    tfrecords_path = gfile.Glob(os.path.join(FLAGS.data_dir, FLAGS.input_file_pattern))
    # tfrecords_path = os.path.join(FLAGS.data_dir, FLAGS.input_file_pattern)
    file_temp_path = FLAGS.output_path
    if not os.path.isdir(file_temp_path):
        os.makedirs(file_temp_path)

    count = 0
    for file_ in tfrecords_path:
        video_name_str = 'validate' + ('%04d' % count)
        tfrecord_name = os.path.join(file_temp_path, video_name_str + '.tfrecords')
        writer = tf.python_io.TFRecordWriter(tfrecord_name)
        record_iterator = tf.python_io.tf_record_iterator(path=file_)
        for string_record in record_iterator:
            example = tf.train.SequenceExample()
            example.ParseFromString(string_record)
            context = {}
            context['id'] = tf.train.Feature(bytes_list=example.context.feature['id'].bytes_list)
            context['labels'] = tf.train.Feature(int64_list=example.context.feature['labels'].int64_list)
            feature = {}
            #print(len(example.feature_lists.feature_list['rgb'].feature))
            feature['rgb'] = tf.train.FeatureList(feature=[x for x in example.feature_lists.feature_list['rgb'].feature])
            feature['audio'] = tf.train.FeatureList(feature=[x for x in example.feature_lists.feature_list['audio'].feature])

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
