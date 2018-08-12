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

    file_temp_path = FLAGS.output_path
    if not os.path.isdir(file_temp_path):
        os.makedirs(file_temp_path)

    # filenames = tfrecords_path[FLAGS.batch_start: min(len(tfrecords_path), FLAGS.batch_end)]
    filename_queue = tf.train.string_input_producer(tfrecords_path, shuffle=False, num_epochs=1)
    reader = tf.TFRecordReader()
    serialized_key, serialized_example = reader.read(filename_queue)
    contexts, features = parse_example_proto(serialized_example)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.local_variables_initializer())
        # sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        csv_content = pandas.read_csv(FLAGS.csv_file)
        num_videos = len(csv_content['VideoId'])
        # assert(num_videos == FLAGS.number_of_videos)

        count = 0
        count_videos = 0
        while True:
            try:
                video_name_str = 'validate' + ('%04d' % count)
                tfrecord_name = os.path.join(file_temp_path, video_name_str + '.tfrecords')
                writer = tf.python_io.TFRecordWriter(tfrecord_name)
                for i in range(290):
                    contexts_dict, features_dict = sess.run([contexts, features])
                    count_videos += 1
                    video_id = contexts_dict['id'].decode("utf-8")
                    csv_object = csv_content.loc[csv_content['VideoId'] == video_id]
                    prediction_pair = csv_object['LabelConfidencePairs'].values[0].split(' ')
                    encoded_predictions = np.zeros(FLAGS.number_of_classes, dtype=np.float32)

                    for prediction_index in range(0, len(prediction_pair) // 2, 2):
                        label_index, label_score = prediction_pair[prediction_index], prediction_pair[prediction_index + 1]
                        encoded_predictions[int(label_index)] = float(label_score)

                    new_contexts_dict = {}
                    new_contexts_dict['id'] = _bytes_feature(contexts_dict['id'])
                    new_contexts_dict['labels'] = _int64_feature(sorted(contexts_dict['labels'].values))
                    new_contexts_dict['ensemble_labels'] = tf.train.Feature(float_list=tf.train.FloatList(value=encoded_predictions))

                    example = tf.train.SequenceExample(context=tf.train.Features(feature=new_contexts_dict))
                    # example = tf.train.Example(features=tf.train.Features(feature=i3d_dict))
                    # example = tf.train.SequenceExample(context=tf.train.Features(feature=contexts_dict),
                    #                                    feature_lists=tf.train.FeaturesLists(feature_lists=features_dict))
                    writer.write(example.SerializeToString())
                writer.close()
                count += 1
                print('Total number of video: %d, processed: %d' % (num_videos, count_videos))
            except tf.errors.OutOfRangeError:
                writer.close()
                break
                count += 1

        # print('Total number of video: %d, processed: %d' % (num_videos, count_videos))
        coord.request_stop()
        coord.join(threads)
        sess.close()


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
