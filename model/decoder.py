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

def embedding_matrix(vocab_size, embedding_dim):
    return variables.model_variable('embedding', shape=(vocab_size, embedding_dim),
                                    initializer=tf.random_normal_initializer(
                                        mean=0.0, stddev=0.01))

def token_prob(ht, context, prev_token_embedding, vocab_size, hparams={}):
    """ Computes probability distribution for next output token.

    Args:
        ht: current hidden state of LSTM
        context: attention-based visual context vector
        prev_token_embedding: embedding of previously emitted token
        vocab_size: cardinality of set of possible output tokens

    Returns:
        (logits, probs) over the set of |vocab_size| tokens.
    """
    hdim = hparams['hdim']
    vdim = hparams['vdim']
    embedding_dim = hparams['embedding_dim']

    with tf.variable_scope("output"):
        L_h = variables.model_variable('L_h', shape=(hdim, embedding_dim),
                                       initializer=tf.random_normal_initializer(
                                            mean=0.0, stddev=np.sqrt(2.0/hdim)))
        L_c = variables.model_variable('L_c', shape=(vdim, embedding_dim),
                                       initializer=tf.random_normal_initializer(
                                            mean=0.0, stddev=np.sqrt(2.0/vdim)))
        L_o = variables.model_variable('L_o', shape=(embedding_dim, vocab_size),
                                       initializer=tf.random_normal_initializer(
                                           mean=0.0,
                                           stddev=np.sqrt(2.0/embedding_dim)))

        # We might want a non-linearity between the matmuls.
        logits = tf.matmul(tf.matmul(ht, L_h) + tf.matmul(context, L_c) + prev_token_embedding, L_o)
        probs = tf.nn.softmax(logits, -1)
        return logits, probs


def decode(feat, lstm, state_tuple, attn_fn, prev_token, embeddings, hparams={}):
    """ Runs a single step of decoding with attention.

    Args:
        feat: tensor of visual features
        lstm: an LSTMCell to use for decoding
        state_tuple: current state of 'lstm' cell, (c_state, m_state)
        attn_fn: a function that returns 'soft-attention' weights and a context
                 vector of same dims as a single visual feature vector

    Returns:
        (LSTM output, new state, attention weights)
    """
    hdim = hparams['hdim']
    vdim = hparams['vdim']
    adim = hparams['adim']
    vocab_size = embeddings.get_shape()[1]
    batch_size = hparams['batch_size']
    c_state, m_state = state_tuple

    attention, context = attn_fn(feat, m_state, hdim=hdim, vdim=vdim, adim=adim,
                                 batch_size=batch_size)

    # TODO(kjchavez):
    # We probably want to loop in an embedded version of the output token from
    # the previous step in the sequence, a la Show, Attend, and Tell.
    # prev_embedding = tf.nn.embedding_lookup
    prev_embedding = tf.squeeze(tf.nn.embedding_lookup(embeddings, prev_token))
    print "Prev embedding shape:", prev_embedding.get_shape()
    print "context shape:", context.get_shape()
    x = tf.concat(1, [prev_embedding, context])
    output, state_tuple = lstm(x, state_tuple)
    logits, probs = token_prob(state_tuple[0], context, prev_embedding, vocab_size,
                               hparams=hparams)
    return probs, state_tuple, attention

