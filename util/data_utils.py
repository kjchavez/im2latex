import cv2
from matplotlib import pyplot as plt
import numpy as np

def formula_size(image):
    """ Returns the patch of image containing non-white pixels, with some padding. """
    mask = np.sum(image, 2) == 0
    horizontal = np.where(np.sum(mask, 0) != 0)[0]
    vertical = np.where(np.sum(mask, 1) != 0)[0]
    width = horizontal[-1] - horizontal[0]
    height = vertical[-1] - vertical[0]
    return height, width


def get_window(image, padding=50):
    """ Returns the padded patch of image containing non-white pixels. """
    mask = np.sum(image, 2) == 0
    horizontal = np.where(np.any(mask, 0))[0]
    left = max(horizontal[0] - padding, 0)
    right = min(horizontal[-1] + padding, image.shape[1])
    vertical = np.where(np.any(mask, 1))[0]
    top = max(vertical[0] - padding, 0)
    bottom = min(vertical[-1] + padding, image.shape[0])
    return image[top:bottom, left:right, :]


def get_horizontal_strip(image, height=200):
    """ Returns a horizontal slab, centered at the equation. """
    mask = np.sum(image, 2) == 0
    vertical = np.where(np.sum(mask, 1) != 0)[0]
    center = (vertical[0] + vertical[-1])/2
    return image[center - height/2 : center + height/2]


def visualize_formula(filename):
    im = get_window(cv2.imread(filename))
    plt.figure(figsize=(17,15))
    plt.imshow(im)
    plt.show()

