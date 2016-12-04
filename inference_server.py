import tensorflow as tf

import cv2
import numpy as np
import argparse
import zmq
from model.att_model import dynamic_model_fn
from model.input import get_train_data
from util.pad import pad_image

tf.logging.set_verbosity(tf.logging.INFO)

config = tf.contrib.learn.RunConfig(save_checkpoints_secs=100)

with open('character_mapping.txt') as fp:
    char_mapping = [l.strip('\n') for l in fp]
    VOCAB_SIZE = len(char_mapping)

params = {
    'hdim': 100,
    'adim': 128,
    'vdim': 256,
    'batch_size': 1,
    'embedding_dim': 80,
    'starter_learning_rate': 0.0005,
    'token_map': char_mapping
}
parser = argparse.ArgumentParser()
parser.add_argument("--address", default="tcp://0.0.0.0:12345")
args = parser.parse_args()

sock = zmq.Context.instance().socket(zmq.REP)
sock.bind(args.address)

def input_fn():
    filename = sock.recv()
    im = cv2.imread(filename)
    h_buckets = [60, 100, 200, 350]
    w_buckets = [400, 800, 1000, 1200, 1400]
    buckets = [(h, w) for h in h_buckets for w in w_buckets]
    im, _ = pad_image(np.all(im == 0, 2).astype(np.float32), buckets, value=0)
    binary_im = tf.constant(im) #np.expand_dims(im - 0.5, 0))
    downsampled = tf.nn.avg_pool(tf.expand_dims(tf.expand_dims(binary_im, 2), 0), [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    downsampled = tf.squeeze(downsampled, [0, 3])
    return {'image': tf.expand_dims(downsampled - 0.5, 0)}

estimator = tf.contrib.learn.Estimator(model_fn=dynamic_model_fn,
                                       model_dir="/tmp/test", config=config,
                                       params=params)


while True:
    print "Prediction"
    print "====================="
    tokens = estimator.predict(input_fn=input_fn)
    print tokens
    chars = []
    for t in tokens[0]:
        ch = char_mapping[t]
        if ch == 'STOP':
            break
        chars.append(ch)
    latex = ''.join(chars)
    print latex
    sock.send(latex)
