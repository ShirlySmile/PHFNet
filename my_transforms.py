
import numpy as np
from PIL import Image
import cv2


class Noprocess(object):

    def __init__(self, r=0):
        self.r = r

    def __call__(self, img):
        return img



def flip(*arrays):
    horizontal = np.random.random() > 0.5
    vertical = np.random.random() > 0.5
    if horizontal:
        arrays = [np.flip(arr, axis=1) for arr in arrays]
    if vertical:
        arrays = [np.flip(arr, axis=2) for arr in arrays]
    return arrays


def rotate(*arrays):
    rotate = np.random.random() > 0.5
    if rotate:
        angle = np.random.choice([1, 2, 3])
        arrays = [np.rot90(arr, k=angle, axes=(1, 2)) for arr in arrays]
    return arrays



