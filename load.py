import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import join
from skimage import data, io

def is_image(filename):
    ext = [".jpg", ".png", ".gif"]
    return any([filename.endswith(e) for e in ext])

def load(pos_dir, neg_dir):
    images, labels = [], []
    files = listdir(pos_dir)
    for f in files:
        if not is_image(f):
            continue
        images.append(data.imread(join(pos_dir, f)))
        labels.append(1)
    files = listdir(neg_dir)
    for f in files:
        if not is_image(f):
            continue
        images.append(data.imread(join(neg_dir, f)))
        labels.append(0)
    return images, labels
