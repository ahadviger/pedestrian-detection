import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import join
from skimage import data, io

def is_image(filename):
    ext = [".jpg", ".png", ".gif"]
    return any([filename.endswith(e) for e in ext])

def load(pos_dir, neg_dir):
    dataset = []
    files = listdir(pos_dir)
    for f in files:
        if not is_image(f):
            continue
        dataset.append((data.imread(join(pos_dir, f)), 1))
    files = listdir(neg_dir)
    for f in files:
        if not is_image(f):
            continue
        dataset.append((data.imread(join(neg_dir, f)), 0))
    return dataset
    
train_pos_dir, train_neg_dir = "train/pos", "train/neg"
test_pos_dir, test_neg_dir = "test/pos", "test/neg"

train = load(train_pos_dir, train_neg_dir)
test = load(test_pos_dir, test_neg_dir)