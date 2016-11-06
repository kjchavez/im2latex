import tensorflow as tf
from model.input import get_train_data
from model.encoder import convolutional_features
from model.decoder import attention
import numpy as np

hdim = 256
adim = 128
batch_size = 2

X, y = get_train_data(batch_size=batch_size)
feat = convolutional_features(tf.expand_dims(X['image'], -1))
h_0 = tf.zeros((batch_size, hdim))
att, context = attention(feat, h_0, hdim=hdim, adim=adim, batch_size=batch_size)
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(init_op)
    feat_val, att_val, c_val = sess.run([feat, att, context])
    coord.request_stop()
    coord.join(threads)

print feat_val.shape
print att_val.shape
print att_val
print np.sum(att_val[0])
print c_val
