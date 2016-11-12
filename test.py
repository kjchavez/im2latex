import tensorflow as tf
from model.input import get_train_data
from model.encoder import convolutional_features
from model.decoder import attention, decode, embedding_matrix
import numpy as np

hparams = {
    'hdim': 256,
    'adim': 128,
    'vdim': 512,
    'batch_size': 2,
    'embedding_dim': 128,
}

with open('character_mapping.txt') as fp:
    char_mapping = [l for l in fp]
    VOCAB_SIZE = len(char_mapping)

embeddings = embedding_matrix(VOCAB_SIZE, hparams['embedding_dim'])
X, y = get_train_data(batch_size=hparams['batch_size'])
feat = convolutional_features(tf.expand_dims(X['image'], -1))
h_0 = tf.zeros((hparams['batch_size'], hparams['hdim']))
c_0 = tf.zeros((hparams['batch_size'], hparams['hdim']))
init_token = tf.zeros((2,1), dtype=tf.int32)
lstm = tf.nn.rnn_cell.BasicLSTMCell(hparams['hdim'])

output, (c1, h1), att = decode(feat, lstm, (c_0, h_0), attention, init_token,
                               embeddings, hparams)

# att, context = attention(feat, h_0, hdim=hdim, adim=adim, batch_size=batch_size)
init_op = tf.initialize_all_variables()


with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(init_op)
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
