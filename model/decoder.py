""" Attention-based decoder from arbitrarily-sized visual features. """
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import variables
import numpy as np

def attention(visual_features, h_t, hdim=256, vdim=512, adim=128, batch_size=2):
    """ Returns a tensor of 'attention' to the given visual_features.

    Args:
        hdim: dimensionality of hidden state of decoder
        vdim: dimensionality of visual features
        adim: dimensionality of hidden layer of attention model

        visual_features: tensor with shape BATCH_SIZE x H x W x VDIM
        h_t:  tensor with shape BATCH_SIZE  x HDIM representing the hidden
              state of the decoder.
    """
    with tf.variable_scope("attention"):
        W_h = variables.model_variable('W_h', shape=(hdim, adim),
                                       initializer=tf.random_normal_initializer(
                                            mean=0.0, stddev=np.sqrt(2.0/hdim)))
        W_v = variables.model_variable('W_v', shape=(vdim, adim),
                                       initializer=tf.random_normal_initializer(
                                            mean=0.0, stddev=np.sqrt(2.0/vdim)))
        B = variables.model_variable('beta', shape=(adim, 1),
                                     initializer=tf.random_normal_initializer(
                                         mean=0.0, stddev=np.sqrt(2.0/adim)))

        # Flatten the 'spatial' dimensions of the visual feature map.
        visual_features = tf.reshape(visual_features, (batch_size, -1, vdim))
        hidden = tf.reshape(tf.matmul(tf.reshape(visual_features, [-1, vdim]), W_v),
                         (batch_size, -1, adim))

        # Here, contribution from the hidden state should be broadcast to all
        # visual features.
        hidden += tf.expand_dims(tf.matmul(h_t, W_h), 1)

        # This is the vector of weights corresponding to how strongly we are
        # 'attending' to any particular visual patch.
        logits = tf.reshape(
                    tf.matmul(tf.reshape(tf.tanh(hidden), (-1, adim)), B),
                    (batch_size, -1))
        att = tf.nn.softmax(logits)
        context = tf.reduce_sum(visual_features * tf.expand_dims(att, -1), 1)
        return att, context
