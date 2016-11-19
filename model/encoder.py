import tensorflow as tf
from tensorflow.contrib.framework.python.ops import variables
import numpy as np

def conv_layer(name, shape, image):
    N = shape[0]*shape[1]*shape[2]
    kernel = variables.model_variable('%s_W' % (name,), shape=shape,
                                      initializer=tf.random_normal_initializer(
                                          mean=0.0, stddev=np.sqrt(2.0/N)))

    tf.add_to_collection(tf.GraphKeys.WEIGHTS, kernel)

    b = variables.model_variable('%s_b' % (name,),
                                 initializer=tf.zeros_initializer([shape[-1]]))

    embedding = tf.nn.bias_add(tf.nn.conv2d(image, kernel, [1, 1, 1, 1],
                                          padding='SAME'), b)
    embedding = tf.nn.relu(embedding)
    return embedding

def max_pool2(x):
    return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

def convolutional_features(image):
    with tf.variable_scope("encoder"):
        f = conv_layer("conv1", [3, 3, 1, 64], image)
        f = max_pool2(f)
        f = conv_layer("conv2", [3, 3, 64, 128], f)
        f = max_pool2(f)
        f = conv_layer("conv3", [3, 3, 128, 256], f)
        f = max_pool2(f)
        f = conv_layer("conv4", [3, 3, 256, 512], f)
        # Note, we may want to leave off the ReLU from the topmost layer.
        return f
