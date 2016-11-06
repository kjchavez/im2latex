""" Zero pads images to match the closest bucket size. """

import argparse
import cv2
import glob
import numpy as np
from multiprocessing import Pool

def find_bucket(image_size, buckets):
    for height, width in buckets:
        if image_size[0] <= height and image_size[1] <= width:
            return (height, width)

def pad_image(image, buckets):
    """ Returns padded image and bucket to which it was padded. """
    bucket = find_bucket(image.shape[0:2], buckets)
    if bucket is None:
        print "Couldn't find bucket."
        return None, None

    vpad = (bucket[0] - image.shape[0])/2
    hpad = (bucket[1] - image.shape[1])/2

    padded_image = cv2.copyMakeBorder(image,
                                      vpad, (bucket[0] - image.shape[0] - vpad),
                                      hpad, (bucket[1] - image.shape[1] - hpad),
                                      cv2.BORDER_CONSTANT,value=(255,255,255))
    return padded_image, bucket

def pad_and_save_image(filename, output_filename, buckets):
    image = cv2.imread(filename)
    if image is None:
        return False

    padded_image, bucket = pad_image(image, buckets)
    if padded_image is None:
        return False

    cv2.imwrite(output_filename, padded_image)
    return True


def pad_images(filenames, output_dir, buckets):
    for f in filenames:
        output_file = os.path.join(output_dir, os.path.basedir(f))
        pad_image(f, output_file, buckets)


def parse_buckets(bucket_str):
    return [eval(b) for b in bucket_str.split(' ')]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepattern", help="filepattern of images to pad.")
    parser.add_argument("--buckets", default="(60, 400) (60, 800) (60, 1000) (60, 1200) (60, 1400) (100, 400) (100, 800) (100, 1000) (100, 1200) (100, 1400) (200, 400) (200, 800) (200, 1000) (200, 1200) (200, 1400) (350, 400) (350, 800) (350, 1000) (350, 1200) (350, 1400)")
    parser.add_argument("--outputdir", '-o', default="output")
    parser.add_argumet("--num_threads", type=int, default=4)
    return parser.parse_args()


def main():
    args = get_args()
    filenames = glob.glob(args.filepattern)

    pool = Pool(args.num_threads)
    pool.map(pad_image, [(f, os.path.join(args.outputdir, os.path.basedir(f)), buckets) for f in filenames])
    pool.join()
