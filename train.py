import tensorflow as tf

import argparse
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
    'epsilon': 1e-8,
    'vdim': 512,
    'batch_size': 10,
    'embedding_dim': 128,
    'starter_learning_rate': 0.0005,
    'unroll_length': 10,
    'token_map': char_mapping
}
parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["train", "infer"], default="train")
parser.add_argument("--image", help="image filename")
args = parser.parse_args()

if args.mode == "train":
    estimator = tf.contrib.learn.Estimator(model_fn=model_fn,
                                           model_dir="/tmp/test", config=config,
                                           params=params)

    monitors = []
    """
    validation_metrics = {"val_accuracy": tf.contrib.metrics.streaming_accuracy}
    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        input_fn=lambda: get_eval_data(batch_size=32), eval_steps=16,
        every_n_steps=1000, metrics=validation_metrics)

    validation_metrics = {"train_acc": tf.contrib.metrics.streaming_accuracy}
    train_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        input_fn=lambda: get_train_data(batch_size=32), eval_steps=68,
        every_n_steps=1000, metrics=validation_metrics)

    grad_monitor = tf.contrib.learn.monitors.SummarySaver(tf.merge_all_summaries(),
                                                          save_steps=100,
                                                          output_dir="/tmp/test")
    monitors = [validation_monitor, train_monitor, grad_monitor]
    """
    estimator.fit(input_fn=lambda: get_train_data(batch_size=params['batch_size']),
                  monitors=monitors, steps=1000000)

elif args.mode == "infer":
    import cv2
    import numpy as np
    im = cv2.imread(args.image)
    def input_fn():
        binary_im = tf.constant(
                        np.expand_dims(
                            np.all(im == 0, 2).astype(np.float32) - 0.5,
                            0))
        return {'image': binary_im}

    params['batch_size'] = 1
    estimator = tf.contrib.learn.Estimator(model_fn=model_fn,
                                           model_dir="/tmp/test", config=config,
                                           params=params)


    print "Prediction"
    print "====================="
    tokens = estimator.predict(input_fn=input_fn)
    print ''.join([char_mapping[t[0]] for t in tokens])
