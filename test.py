import tensorflow as tf
from model.input import get_train_data
from model.encoder import convolutional_features

X, y = get_train_data(batch_size=2)
f = convolutional_features(tf.expand_dims(X['image'], -1))
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(init_op)
    image_val = sess.run(f)
    print image_val
    coord.request_stop()
    coord.join(threads)

