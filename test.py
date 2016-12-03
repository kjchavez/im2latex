import tensorflow as tf
from model.input import get_train_data
from model.encoder import convolutional_features
from model.decoder import *
from model.att_model import *
import numpy as np

hparams = {
    'hdim': 128,
    'adim': 128,
    'vdim': 512,
    'batch_size': 2,
    'embedding_dim': 128,
    'output_feedback': True,
}

with open('character_mapping.txt') as fp:
    char_mapping = [l for l in fp]
    VOCAB_SIZE = len(char_mapping)

embeddings = embedding_matrix(VOCAB_SIZE, hparams['embedding_dim'])
X, y = get_train_data(batch_size=hparams['batch_size'])
feat = convolutional_features(tf.expand_dims(X['image'], -1))
h_0 = tf.zeros((hparams['batch_size'], hparams['hdim']))
c_0 = tf.zeros((hparams['batch_size'], hparams['hdim']))
init_token = tf.zeros((2,), dtype=tf.int32)
lstm = tf.nn.rnn_cell.BasicLSTMCell(hparams['hdim'])

logits, output, (c1, h1), att = decode(feat, lstm, (c_0, h_0), attention, init_token,
                               embeddings, hparams)

# Make 'y' time major.
print y['target'].get_shape()
print y['sequence_length'].get_shape()
input_ = tf.to_int64(array_ops.transpose(y['target'], [1, 0]))
#fn = get_loop_fn(input_, y['sequence_length'], feat, lstm, (c_0, h_0), attention, init_token,
#                 embeddings, hparams)

initial_state = lstm.zero_state(hparams['batch_size'], tf.float32)
print initial_state[1].get_shape()

fn = get_loop_fn(input_, y['sequence_length'], feat, lstm,
                 lstm.zero_state(hparams['batch_size'], tf.float32), attention,
                 init_token, embeddings, hparams)

emit_ta, final_state, final_loop_state = tf.nn.raw_rnn(lstm, fn)

# att, context = attention(feat, h_0, hdim=hdim, adim=adim, batch_size=batch_size)
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(init_op)
    final_state_val, seq = sess.run([final_state, y['sequence_length']])
    coord.request_stop()
    coord.join(threads)

print final_state_val
print seq

"""
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(init_op)
    y_val = sess.run(y)
    print "Target:", y_val['target']
    print "Weights:", y_val['weights']
    print y_val['target'].shape
    print y_val['weights'].shape
    feat_val, att_val, out_val = sess.run([feat, att, output])
    coord.request_stop()
    coord.join(threads)

print feat_val.shape
print att_val.shape
print out_val

ML = np.argmax(out_val, 1)
print ML
print char_mapping[ML[0]]
print char_mapping[ML[1]]
"""
