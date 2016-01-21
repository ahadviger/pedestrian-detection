import cPickle
import argparse
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

parser = argparse.ArgumentParser(description='Processes images and detects pedestrians.')
parser.add_argument('command', choices=['train_model', 'prepare_initial', 'prepare_hard_examples'])
parser.add_argument('--descriptor', choices=['hog', 'css', 'hogcss'], required=True)
parser.add_argument('--model')
parser.add_argument('--prefix')
parser.add_argument('--hard_examples', nargs='*')
args = parser.parse_args()

if args.descriptor == "hog":
    descriptor = HOG()
elif args.descriptor == "css":
    descriptor = CSS()
elif args.descriptor == "hogcss":
    descriptor = HOGCSS()
model = Model(descriptor)

if args.command == 'prepare_initial':
    model.prepare_initial()
elif args.command == 'prepare_hard_examples':
    model.load(args.model)
    model.prepare_hard_examples(args.prefix)
elif args.command == 'train_model':
    print "Loading training and test set..."
    images_train, labels_train = load(WINDOW_TRAIN_POS, WINDOW_TRAIN_NEG, args.hard_examples)
    images_test, labels_test = load(WINDOW_TEST_POS, WINDOW_TEST_NEG, args.hard_examples)

    print "Loaded {0} train and {1} test images".format(len(images_train), len(images_test))
    model.train(images_train, labels_train, images_test, labels_test)
    model.save(args.model)