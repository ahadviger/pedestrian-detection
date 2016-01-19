import cPickle

import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import join
from skimage import data, io, color
from skimage.feature import hog
from sklearn.svm import SVC

from load import *
 
from sklearn.metrics import confusion_matrix

from model import *
from descriptor import *

descriptor = HOG()
model = Model(descriptor)
#model.prepare_initial()
model.load('hog_step3')
model.prepare_hard_negative('hard_negative_6_')

if True:
    print "Loading training and test set..."
    images_train, labels_train = load(WINDOW_TRAIN_POS, WINDOW_TRAIN_NEG)
    images_test, labels_test = load(WINDOW_TEST_POS, WINDOW_TEST_NEG)

    model.train(images_train, labels_train, images_test, labels_test)
    model.save('hog_step4')

if False:
    train_pos_dir, train_neg_dir = "train/pos", "train/neg"
    test_pos_dir, test_neg_dir = "test/pos", "test/neg"

    print "Loading dataset..."
    train, labels_train = load(train_pos_dir, train_neg_dir)
    test, labels_test = load(test_pos_dir, test_neg_dir)

    print "Calculating HOG features..."
    hog_train, hog_test = [], []

    for image in train:
        hog_train.append(hog(color.rgb2gray(image), orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualise=False))
                        
    for image in test:
        hog_test.append(hog(color.rgb2gray(image), orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualise=False))

    print "Training..."
    model = SVC()
    model.fit(hog_train, labels_train)

    print "Train score:", model.score(hog_train, labels_train)
    print "Test score:", model.score(hog_test, labels_test)

    #for image, hog_input, label in zip(train, hog_train, labels_train):
    #    if (model.predict([hog_input])[0] != label):
    #        print(label)
    #        plt.imshow(image)
    #        plt.show()

    print(confusion_matrix(labels_train, model.predict(hog_train)))
    print(confusion_matrix(labels_test, model.predict(hog_test)))

    with open('classifier.pkl', 'wb') as fid:
        cPickle.dump(model, fid)
