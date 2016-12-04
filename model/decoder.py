""" Attention-based decoder from arbitrarily-sized visual features. """
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.ops import array_ops
import numpy as np

from .input import STOP_ID

def attention(visual_features, h_t, hdim=256, vdim=512, adim=128,
              batch_size=2):
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

def get_loop_fn(decoder_inputs, sequence_length, feat, lstm, initial_state,
                attn_fn, init_token, embeddings, hparams={}):
    """ Returns a loop function that can be used in the 'raw_rnn' function.

    Args:
        decoder_inputs: time-major decoder inputs
        ...
    """
    input_shape = array_ops.shape(decoder_inputs)
    time_steps = input_shape[0]
    decoder_inputs_ta = tf.TensorArray(dtype=decoder_inputs.dtype, size=time_steps)
    decoder_inputs_ta = decoder_inputs_ta.unpack(decoder_inputs)
    vocab_size = embeddings.get_shape()[0]

    def loop_fn(time, cell_output, cell_state, prev_token_embedding):
        """
        The loop_state is the embedding for the previous token.
        """
        with tf.variable_scope("decoder"): #, reuse=reuse_vars):
            # If it's the initial iteration, there is some special setup to do.
            if cell_output is None:
                # TODO(kjchavez): It would be cleaner to instantiate all model
                # variables outside the loop function and just use them here.
                elements_finished = (time >= sequence_length)
                next_cell_state = initial_state
                selected_token_embedding = tf.nn.embedding_lookup(embeddings,
                                                                  init_token)
                print "TOKEN EMBEDDING SHAPE:", selected_token_embedding.get_shape()
                attention, context = attn_fn(feat, initial_state[1], hdim=hparams['hdim'],
                                             vdim=hparams['vdim'], adim=hparams['adim'],
                                             batch_size=hparams['batch_size'])

                # Ignored on first iteration, but used to setup graph.
                logits, probs = token_prob(initial_state[1], context,
                                           selected_token_embedding, vocab_size,
                                           hparams=hparams)

                next_input = tf.concat(1, [selected_token_embedding, context])
                return (elements_finished, next_input, next_cell_state,
                        None, selected_token_embedding)

            # All subsequent iterations
            tf.get_variable_scope().reuse_variables()
            next_cell_state = cell_state

            # Given the cell output, the emit output is a distribution over tokens.
            attention, context = attn_fn(feat, cell_output, hdim=hparams['hdim'],
                                         vdim=hparams['vdim'], adim=hparams['adim'],
                                         batch_size=hparams['batch_size'])
            logits, probs = token_prob(cell_output, context, prev_token_embedding, vocab_size,
                                       hparams=hparams)
            emit_output = logits
            print "Emit output shape:", emit_output.get_shape()

            # To produce the next input from the current output, we must do a
            # couple of things. First, we choose a token from the token
            # distribution (or use the 'true' token).
            if hparams['output_feedback']:
                selected_token = tf.argmax(logits, 1, name="argmax_token")
            else:
                # We should be careful not to read past the last token.
                # Since we are using time - 1, this implies that the
                # decoder_inputs should NOT contain the GO token.
                selected_token = decoder_inputs_ta.read(time-1)

            selected_token_embedding = tf.nn.embedding_lookup(embeddings, selected_token)
            print "selected token embedding:", selected_token_embedding

            elements_finished = tf.logical_or(time >= sequence_length,
                                              selected_token == STOP_ID)
            finished = tf.reduce_all(elements_finished)

            next_input = tf.cond(
                finished,
                lambda: tf.zeros([hparams['batch_size'],
                                  hparams['embedding_dim'] +hparams['vdim']], dtype=tf.float32),
                lambda: tf.concat(1, [selected_token_embedding, context]))

        return (elements_finished, next_input, next_cell_state,
                emit_output, selected_token_embedding)


    return loop_fn

def decode(feat, lstm, state_tuple, attn_fn, prev_token, embeddings,
           hparams={}):
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

    #prev_embedding = tf.squeeze(tf.nn.embedding_lookup(embeddings, prev_token))
    prev_embedding = tf.nn.embedding_lookup(embeddings, prev_token)
    print "Prev embedding:", prev_embedding.get_shape()
    print "Context:", context.get_shape()
    x = tf.concat(1, [prev_embedding, context])
    output, state_tuple = lstm(x, state_tuple)
    logits, probs = token_prob(output, context, prev_embedding, vocab_size,
                               hparams=hparams)
    return logits, probs, state_tuple, attention
