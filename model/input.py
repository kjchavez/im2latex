import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import parsing_ops

train_filename = "/home/kevin/projects/im2latex/im2latex_train.tfrecord"
dev_filename = "/home/kevin/projects/im2latex/im2latex_dev.tfrecord"

PAD_ID = 0
GO_ID = 1
STOP_ID = 2

def extract_image(image_placeholder):
    """ Creates the preprocessed input from image_placeholder.

    Args:
        image_placeholder: a placeholder with filenames of size (batch_size, 1)
    """
    images = [tf.image.decode_png(tf.read_file(image_placeholder[i]),
                                  channels=1)
              for i in xrange(image_placeholder.get_shape()[0])]

    # Pad images so that they may be batched together.

    return image_data


def preprocess(image):
    """ Downsample and center around zero. """
    downsampled = tf.nn.avg_pool(tf.expand_dims(tf.expand_dims(image, 2), 0), [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    downsampled = tf.squeeze(downsampled, [0, 3])
    return downsampled - 0.5

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
                "sequence_length": tf.FixedLenFeature([], tf.int64),
                "label": tf.VarLenFeature(tf.int64)
            })

        label = tf.cast(features.pop('label'), tf.int32)
        sequence_length = tf.cast(features.pop('sequence_length'), tf.int32)

        image = tf.cast(tf.decode_raw(features['image_raw'], tf.uint8),
                         tf.float32)

        shape = tf.cast(tf.concat(0, [tf.reshape(features['height'], (1,)),
                              tf.reshape(features['width'], (1,))]), tf.int32)

        image = preprocess(tf.reshape(image, shape))

        image, label, sequence_length = tf.train.batch(
                                            [image, label, sequence_length],
                                            batch_size=batch_size,
                                            capacity=200, dynamic_pad=True,
                                            num_threads=4)

    with tf.variable_scope("target_sequence"):
        label = tf.reshape(tf.sparse_to_dense(label.indices, label.shape, label.values,
                                   default_value=PAD_ID), (batch_size, -1),
                           name="tokens")
        weights = tf.reshape(tf.cast(tf.not_equal(label, PAD_ID), tf.float32),
                             (batch_size, -1), name="weights")

    return {'image': image}, {'target': label, 'weights': weights,
                              'sequence_length': sequence_length}

def get_train_data(batch_size=1):
    return get_feature_input(train_filename, batch_size=batch_size)

def get_eval_data(batch_size=1):
    return get_feature_input(dev_filename, batch_size=batch_size)
