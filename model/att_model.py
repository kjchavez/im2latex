import tensorflow as tf
from tensorflow.contrib.framework.python.ops import variables
import math
from .encoder import convolutional_features
from .decoder import decode, attention

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
        return (initial_hidden, initial_memory)


class LSTMSingleton(object):
    lstm = None

    @staticmethod
    def get_instance(dim):
        if LSTMSingleton.lstm is None:
            LSTMSingleton.lstm = tf.nn.rnn_cell.BasicLSTMCell(dim)
        return LSTMSingleton.lstm

def start_sequence_token(batch_size):
    return tf.ones(batch_size, dtype=tf.int32)

def model_fn(features, targets, mode, params):
    hparams = params
    VOCAB_SIZE = len(hparams['token_map'])
    embeddings = embedding_matrix(VOCAB_SIZE, hparams['embedding_dim'])
    image = features['image']
    feat = convolutional_features(tf.expand_dims(image, -1))

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
