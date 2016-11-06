import argparse
import glob
import cv2
import os
import numpy as np
from data_utils import get_window

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="filepattern for images to process")
    parser.add_argument("--output_directory", '-o', default="output")
    return parser.parse_args()

def main():
    args = get_args()
    if not os.path.isdir(args.output_directory):
        os.makedirs(args.output_directory)

    for i, filename in enumerate(glob.glob(args.input)):
        image = cv2.imread(filename)
        if image is None:
            continue

        # Otherwise crop and save to output directory.
        try:
            cropped = get_window(image, padding=8)
        except:
            print "Failed to process file:", filename
            continue

        saveto = os.path.join(args.output_directory,
                              os.path.basename(filename))

        cv2.imwrite(saveto, cropped)
        if (i + 1) % 1000 == 0:
            print "Processed %d files." % (i+1)

if __name__ == "__main__":
    main()
