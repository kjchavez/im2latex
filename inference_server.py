import tensorflow as tf

import cv2
import numpy as np
import argparse
import zmq
from model.att_model import model_fn
from model.input import get_train_data

tf.logging.set_verbosity(tf.logging.INFO)

config = tf.contrib.learn.RunConfig(save_checkpoints_secs=100)

with open('character_mapping.txt') as fp:
    char_mapping = [l.strip('\n') for l in fp]
    VOCAB_SIZE = len(char_mapping)

params = {
    'hdim': 256,
    'adim': 128,
    'vdim': 512,
    'batch_size': 1,
    'embedding_dim': 128,
    'starter_learning_rate': 0.0005,
    'unroll_length': 10,
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
    binary_im = tf.constant(
                    np.expand_dims(
                        np.all(im == 0, 2).astype(np.float32) - 0.5,
                        0))
    return {'image': binary_im}

estimator = tf.contrib.learn.Estimator(model_fn=model_fn,
                                       model_dir="/tmp/test", config=config,
                                       params=params)


while True:
    print "Prediction"
    print "====================="
    tokens = estimator.predict(input_fn=input_fn)
    latex = ''.join([char_mapping[t[0]] for t in tokens])
    sock.send(latex)