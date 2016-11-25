import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import parsing_ops

train_filename = "/home/kevin/projects/im2latex/im2latex_train.tfrecord"
dev_filename = "/home/kevin/projects/im2latex/im2latex_dev.tfrecord"

PAD_ID = 0
GO_ID = 1
STOP_ID = 2

def preprocess(image):
    """ Downsample and center around zero. """
    return image - 0.5

def get_feature_input(filepattern, batch_size=1):
    with tf.variable_scope("DataLoader"):
        filename_queue = \
            tf.train.string_input_producer(tf.matching_files(filepattern))
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features = {
                "image_raw": tf.FixedLenFeature([], tf.string),
                "height": tf.FixedLenFeature([], tf.int64),
                "width": tf.FixedLenFeature([], tf.int64),
                "label": tf.VarLenFeature(tf.int64)
            })

        label = tf.cast(features.pop('label'), tf.int32)

        image = tf.cast(tf.decode_raw(features['image_raw'], tf.uint8),
                         tf.float32)

        shape = tf.cast(tf.concat(0, [tf.reshape(features['height'], (1,)),
                              tf.reshape(features['width'], (1,))]), tf.int32)

        image = preprocess(tf.reshape(image, shape))

        image, label = tf.train.batch([image, label],
                                      batch_size=batch_size,
                                      capacity=200,
                                      dynamic_pad=True,
                                      num_threads=4)

    with tf.variable_scope("target_sequence"):
        label = tf.reshape(tf.sparse_to_dense(label.indices, label.shape, label.values,
                                   default_value=PAD_ID), (batch_size, -1),
                           name="tokens")
        weights = tf.reshape(tf.cast(tf.not_equal(label, PAD_ID), tf.float32),
                             (batch_size, -1), name="weights")

    return {'image': image}, {'target': label, 'weights': weights}

def get_train_data(batch_size=1):
    return get_feature_input(train_filename, batch_size=batch_size)

def get_eval_data(batch_size=1):
    return get_feature_input(dev_filename, batch_size=batch_size)
