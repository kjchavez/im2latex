import tensorflow as tf
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.ops import array_ops
import numpy as np
import math
from .encoder import convolutional_features
from .decoder import decode, attention, get_loop_fn
from .input import GO_ID

# Max length of a sequence to decode during inference time.
MAX_LENGTH = 200

def embedding_matrix(vocab_size, embedding_dim):
    return variables.model_variable('embedding', shape=(vocab_size, embedding_dim),
                                    initializer=tf.random_uniform_initializer(-1.0,
                                                                              1.0))

def init_stddev(target_stddev, in_dim):
    return target_stddev / math.sqrt(in_dim)

def get_initial_lstm_state(mean_context, hparams):
    with tf.variable_scope("lstm_init"):
        W_h = variables.model_variable('init_h_W',
                                       shape=(hparams['vdim'], hparams['hdim']),
                                       initializer=tf.random_normal_initializer(
                                           mean=0.0,
                                           stddev=init_stddev(2, hparams['vdim'])))
        b_h = variables.model_variable('init_h_b',
                                       initializer=tf.zeros_initializer(hparams['hdim']))
        W_c = variables.model_variable('init_c_W',
                                       shape=(hparams['vdim'], hparams['hdim']),
                                       initializer=tf.random_normal_initializer(
                                           mean=0.0,
                                           stddev=init_stddev(2, hparams['vdim'])))
        b_c = variables.model_variable('init_c_b',
                                       initializer=tf.zeros_initializer(hparams['hdim']))

        initial_hidden = tf.nn.tanh(tf.matmul(mean_context, W_h) + b_h)
        initial_memory = tf.nn.tanh(tf.matmul(mean_context, W_c) + b_c)
        return tf.nn.rnn_cell.LSTMStateTuple(initial_hidden, initial_memory)


class LSTMSingleton(object):
    lstm = None

    @staticmethod
    def get_instance(dim):
        if LSTMSingleton.lstm is None:
            LSTMSingleton.lstm = tf.nn.rnn_cell.BasicLSTMCell(dim)
        return LSTMSingleton.lstm

def start_sequence_token(batch_size):
    assert GO_ID == 1
    return tf.ones(batch_size, dtype=tf.int32)

def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.
  Args:
    logits: 3D Tensor of shape [batch_size x T x num_decoder_symbols].
    targets: 2D int32 Tensor of shape [batch_size x T]
    weights: 2D float Tensor of the shape as targets
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, defaults to "sequence_loss".
  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).
  """
  with tf.name_scope(name, "sequence_loss", [logits, targets, weights]):
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
    if average_across_timesteps:
        crossent = tf.reduce_mean(crossent*weights, 1)

    if average_across_batch:
      crossent = tf.reduce_mean(crossent, 0)

    return crossent

def dynamic_model_fn(features, targets, mode, params):
    VOCAB_SIZE = len(params['token_map'])
    embeddings = embedding_matrix(VOCAB_SIZE, params['embedding_dim'])
    image = features['image']
    visual_features = convolutional_features(tf.expand_dims(image, -1),
                                             params['vdim'])
    lstm = LSTMSingleton.get_instance(params['hdim'])

    if mode is tf.contrib.learn.ModeKeys.TRAIN:
        target_tokens = targets['target']
        # Decoder inputs should be in time-major order.
        decoder_inputs = tf.to_int64(array_ops.transpose(target_tokens, [1, 0]))
        sequence_length = targets['sequence_length']
        feed_prev_output = params['feed_prev_output']
    else:
        # Doesn't matter how long the sequence is since we will be feeding in
        # tokens predicted from the previous iteration.
        decoder_inputs = tf.zeros((1, params['batch_size']), dtype=tf.int64)
        sequence_length = tf.constant(np.full((params['batch_size'],),
                                              MAX_LENGTH), dtype=tf.int32)
        feed_prev_output = True

    # Builds the actual model.
    initial_state = get_initial_lstm_state(
        tf.reduce_mean(visual_features, (1,2)), params)
    init_token = start_sequence_token(params['batch_size'])
    fn = get_loop_fn(decoder_inputs, sequence_length, visual_features, lstm,
                     initial_state, attention, init_token, embeddings,
                     feed_prev_output=feed_prev_output,
                     hparams=params)
    token_logits, final_state, _ = tf.nn.raw_rnn(lstm, fn)

    # 'token_logits' holds the log-probability of each token for each
    # iteration of the decoding sequence.

    # Get logits as packed tensor in batch-major order
    token_logits = array_ops.transpose(token_logits.pack(), [1, 0, 2])

    if mode is tf.contrib.learn.ModeKeys.TRAIN:
        loss = sequence_loss(token_logits, targets['target'],
                             targets['weights'])

        global_step = variables.get_global_step()
        learning_rate = tf.train.exponential_decay(
                            params['starter_learning_rate'],
                            global_step, 20000,
                            0.96, staircase=True)

        optimizer = tf.train.AdamOptimizer(params['starter_learning_rate'],
                                           epsilon=params['epsilon'])
        train_op = optimizer.minimize(loss, global_step=global_step)

        return (None, loss, train_op)

    else:
        predicted_tokens = tf.argmax(token_logits, 2)
        print "pred tokens shape:", predicted_tokens.get_shape()
        return (predicted_tokens, None, None)

def model_fn(features, targets, mode, params):
    # TODO(kjchavez): Consider using 'embedding_attention_decoder' from
    # seq2seq.py
    hparams = params
    VOCAB_SIZE = len(hparams['token_map'])
    embeddings = embedding_matrix(VOCAB_SIZE, hparams['embedding_dim'])
    image = features['image']
    feat = convolutional_features(tf.expand_dims(image, -1), hparams['vdim'])

    if mode is tf.contrib.learn.ModeKeys.TRAIN:
        target_token_seq = targets['target']
        weights = targets['weights']
        print "Weights:", weights.get_shape()
        print "Targets:", target_token_seq.get_shape()

    lstm = LSTMSingleton.get_instance(hparams['hdim'])

    h, c = get_initial_lstm_state(tf.reduce_mean(feat, (1,2)), params)

    init_token = start_sequence_token(hparams['batch_size'])
    prev_token = init_token
    logits_seq = []
    targets_seq = []
    weights_seq = []
    attention_seq = []
    predicted_tokens = []

    # TODO(kjchavez): During 'inference' time, we actually want to decode until
    # we get a STOP token.
    with tf.variable_scope("decoder"):
        for i in xrange(hparams['unroll_length']):
            if (i > 0): tf.get_variable_scope().reuse_variables()
            print "Iteration %d" % i

            logits, probs, (c, h), att = decode(feat, lstm, (c, h), attention,
                                                prev_token, embeddings,
                                                hparams)

            # During training, we will use the *correct* token at every step in the
            # sequence, but during eval / inference, we use argmax over
            # distribution.
            #
            # TODO(kjchavez): We likely want to perform a beam search over possible
            # sequences at inference time. Not sure if that fits well into the
            # "estimator" model in TF.
            if mode is tf.contrib.learn.ModeKeys.TRAIN:
                logits_seq.append(logits)

                # This is not actually *that* straightforward. Will need to revise.
                targets_seq.append(target_token_seq[:, i])
                weights_seq.append(weights[:, i])
                attention_seq.append(att)

                if hparams['output_feedback']:
                    prev_token = tf.argmax(logits, 1, name="argmax_token")
                else:
                    prev_token = target_token_seq[:, i]
            else:
                prev_token = tf.argmax(logits, 1, name="argmax_token")
                predicted_tokens.append(prev_token)

    # Track a few important distributions.
    if mode is tf.contrib.learn.ModeKeys.TRAIN:
        focus = tf.concat(0, [tf.argmax(att, 1) for
                              att in attention_seq])
        tf.histogram_summary("focus", focus)

        pred_token = tf.concat(0, [tf.argmax(logits, 1) for
                              logits in logits_seq])
        tf.histogram_summary("pred_token", pred_token)

    # Now add losses!
    if mode is tf.contrib.learn.ModeKeys.TRAIN:
        loss = tf.nn.seq2seq.sequence_loss(logits_seq, targets_seq,
                                           weights_seq)

        global_step = variables.get_global_step()
        learning_rate = tf.train.exponential_decay(
                            hparams['starter_learning_rate'],
                            global_step, 20000,
                            0.96, staircase=True)

        # TODO(kjchavez): It's been suggested that ADAM might be a good
        # optimizer for this problem.
        optimizer = tf.train.AdamOptimizer(hparams['starter_learning_rate'],
                                           epsilon=hparams['epsilon'])
        grads_and_vars = optimizer.compute_gradients(loss,
                                                     tf.trainable_variables())

        gradients = tf.gradients(loss, tf.trainable_variables())
        for gv in grads_and_vars:
            tf.histogram_summary("%s_grad" % gv[0].name, gv[0])

        train_op = optimizer.apply_gradients(grads_and_vars,
                                             global_step=global_step)


        # train_op = optimizer.minimize(loss, global_step=global_step)
        tf.scalar_summary("train_loss", loss)
        return (None, loss, train_op)
    else:
        # Predicted tokens -> string?
        return (predicted_tokens, None, None)
