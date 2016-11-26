import argparse
import cv2
import collections
import os
import numpy as np

def find_index_above(values, n):
    """ Given a list of monotonically increasing values, finds the first that
    is above 'n'.
    """
    for i, val in enumerate(values):
        if val > n:
            return i

    return None

class TrainingSet(object):
    def __init__(self, image_dir, formulas, example_list, num_buckets=5):
        self.image_dir = image_dir
        self.formulas = formulas
        self.characters = collections.Counter()
        self.buckets = self.parse(example_list, num_buckets=num_buckets)
        self.keys = sorted(self.buckets.keys())

    def filename(self, basename):
        return os.path.join(self.image_dir, basename) + ".png"

    def get_bucket(self, bucket_id):
        return self.buckets[self.keys[bucket_id]]

    def get_batch(self, bucket_id, batch_size):
        if batch_size > len(self.get_bucket(bucket_id)):
            print "Error. Bucket is too small!"
            return None

        return np.random.choice(self.get_bucket(bucket_id), batch_size,
                                replace=False)

    def parse(self, example_list, num_buckets=5):
        partitioned = {}
        for line in example_list:
            formula_idx, basename, _ = line.split()
            formula = self.formulas[int(formula_idx)]
            imagefile = self.filename(basename)
            image = cv2.imread(imagefile)
            if image is None:
                print "Error reading: %s" % imagefile
                continue

            example = {"filename": imagefile,
                       "formula": formula,
                       "size": image.shape[0:2]}

            length = len(formula)
            if length not in partitioned:
                partitioned[length] = []

            partitioned[length].append(example)
            self.characters.update(formula)

        keys = sorted(partitioned.keys())
        counts = [len(partitioned[key]) for key in keys]
        count = sum(counts)
        bucket_capacity = count / num_buckets
        print "Bucket Capacity =", bucket_capacity
        print "Num diff. lengths = ", len(counts)

        cumulative = [sum(counts[0:(i+1)]) for i in xrange(len(counts))]
        prev_idx = 0
        buckets = {}
        for i in xrange(1, num_buckets):
            idx = find_index_above(cumulative, bucket_capacity*i)
            buckets[keys[idx]] = sum((partitioned[keys[j]] for j in \
                                     xrange(prev_idx, idx+1)), [])
            prev_idx = idx + 1

        # The last boundary is simply the maximum.
        buckets[keys[-1]] = sum((partitioned[keys[j]] for j in \
                                xrange(prev_idx, len(keys))), [])

        for key in buckets:
            print "Bucket=%d, count=%d" % (key, len(buckets[key]))

        return buckets

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", '-i', default="data/cropped_images")
    parser.add_argument("--formulas_file", '-f',
                        default="data/im2latex_formulas.lst")
    parser.add_argument("--train_examples",  default="data/im2latex_train.lst")
    parser.add_argument("--max_train", type=int, default=1000)
    parser.add_argument("--dev_examples",  default="data/im2latex_validate.lst")
    parser.add_argument("--max_dev", type=int, default=100)
    return parser.parse_args()

def examples_generator(examples_file, max_count):
    with open(examples_file) as fp:
        for i, line in enumerate(fp):
            if i < max_count:
                yield line

def main():
    args = get_args()
    with open(args.formulas_file) as fp:
        formulas = [f.strip() for f in fp]

    dataset = TrainingSet(args.image_dir, formulas, examples_generator(args.train_examples, args.max_train))

    print dataset.characters
    print dataset.buckets.keys()
    print dataset.get_batch(0, 10)

if __name__ == "__main__":
    main()
