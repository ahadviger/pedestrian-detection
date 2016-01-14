import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import join
from skimage import data, io, color
from skimage.feature import hog
from sklearn.svm import SVC

from load import *
    
train_pos_dir, train_neg_dir = "train/pos", "train/neg"
test_pos_dir, test_neg_dir = "test/pos", "test/neg"

print "Loading dataset..."
train, labels_train = load(train_pos_dir, train_neg_dir)
test, labels_test = load(test_pos_dir, test_neg_dir)

print "Calculating HOG features..."
hog_train, hog_test = [], []

for image in train:
    hog_train.append(hog(color.rgb2gray(image), orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualise=False))
                    
for image in test:
    hog_test.append(hog(color.rgb2gray(image), orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualise=False))

print "Training..."
model = SVC()
model.fit(hog_train, labels_train)

print "Train score:", model.score(hog_train, labels_train)
print "Test score:", model.score(hog_test, labels_test)