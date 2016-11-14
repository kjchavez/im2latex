import tensorflow as tf
from tensorflow.contrib.framework.python.ops import variables
from .encoder import convolutional_features
from .decoder import decode, attention

def embedding_matrix(vocab_size, embedding_dim):
    return variables.model_variable('embedding', shape=(vocab_size, embedding_dim),
                                    initializer=tf.random_normal_initializer(
                                        mean=0.0, stddev=0.01))

class LSTMSingleton(object):
    lstm = None

    @staticmethod
    def get_instance(dim):
        if LSTMSingleton.lstm is None:
            LSTMSingleton.lstm = tf.nn.rnn_cell.BasicLSTMCell(dim)
        return LSTMSingleton.lstm

def start_sequence_token(batch_size):
    return tf.zeros(batch_size, dtype=tf.int32)

def model_fn(features, targets, mode, params):
    hparams = params
    VOCAB_SIZE = len(hparams['token_map'])
    embeddings = embedding_matrix(VOCAB_SIZE, hparams['embedding_dim'])
    image = features['image']
    feat = convolutional_features(tf.expand_dims(image, -1))
    target_token_seq = targets

    lstm = LSTMSingleton.get_instance(hparams['hdim'])

    # TODO(kjchavez): Compute these based on the average of the conv. features.
    h = tf.zeros((hparams['batch_size'], hparams['hdim']))
    c = tf.zeros((hparams['batch_size'], hparams['hdim']))

    init_token = start_sequence_token(hparams['batch_size'])
    prev_token = init_token
    logits_seq = []
    targets_seq = []

    # TODO(kjchavez): During 'inference' time, we actually want to decode until
    # we get a STOP token.
    with tf.variable_scope("decoder"):
        for i in xrange(hparams['unroll_length']):
            if (i > 0): tf.get_variable_scope().reuse_variables()

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

                prev_token = target_token_seq[:, i]
            else:
                prev_token = tf.argmax(logits, 1, name="argmax_token")

    # Now add losses!
    if mode is tf.contrib.learn.ModeKeys.TRAIN:
        weights = [tf.ones(hparams['batch_size'])]*hparams['unroll_length']
        loss = tf.nn.seq2seq.sequence_loss(logits_seq, targets_seq, weights)

        global_step = variables.get_global_step()
        learning_rate = tf.train.exponential_decay(
                            hparams['starter_learning_rate'],
                            global_step, 20000,
                            0.96, staircase=True)

        # TODO(kjchavez): It's been suggested that ADAM might be a good
        # optimizer for this problem.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        # Need to add summary for GRADIENTS!!
        train_op = optimizer.minimize(loss, global_step=global_step)
        tf.scalar_summary("train_loss", loss)
        return (None, loss, train_op)
    else:
        return (targets_seq, None, None)
