"""
    Commandline utility for inspecting TFRecords.

    Sample usage:

        python tfrecord_viewer.py filename.tfrecord --count 2 \
            --suppress image_raw
"""
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="tfrecord to view")
parser.add_argument("--count", '-c', type=int, default=10,
                    help="number of records to show.")
parser.add_argument("--suppress", type=str, default="",
                    help="comma-separated list of features to suppress"
                         " (usually to big to be useful in terminal)")
args = parser.parse_args()

it = tf.python_io.tf_record_iterator(args.filename)
suppress = [f.strip() for f in args.suppress.split(',')]

example_proto = tf.train.Example()
for i in xrange(args.count):
    tfrecord = next(it)
    example_proto.ParseFromString(tfrecord)
    for f in suppress:
        if f in example_proto.features.feature:
            del example_proto.features.feature[f]
    print "Example #%d" % i
    print "-"*80
    print example_proto

