import cPickle

import scipy
import argparse

from load import *
from skimage import data, io, color
from skimage.transform import pyramid_gaussian, pyramid_expand
from skimage.feature import hog
from sklearn.svm import SVC
from PIL import Image, ImageDraw
from os.path import join, splitext, exists, basename
from os import makedirs

from constants import *
from model import *
from descriptor import *
from joblib import Parallel, delayed
from datetime import datetime

def annotate(image, annotations):
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)

    for [x1, y1, x2, y2] in annotations:
        draw.rectangle([(x1, y1), (x2, y2)])

    return np.asarray(pil)

def process(model, path, pair, i, n):
    start = datetime.now()

    (image_name, image), annotations = pair
    print "[{0}/{1}] Detecting: {2}".format(i, n, image_name)

    detected = model.detect(image)
    minimal = model.non_maximum_suppression(detected)
    end = datetime.now()

    print "Detection time: " + str(end - start)

    true_annotated = annotate(image, annotations)
    all_detected_annotated = annotate(image, detected)
    detected_annotated = annotate(image, minimal)

    true_file_name = splitext(basename(image_name))[0] + '_true' + splitext(image_name)[1]
    all_detected_file_name = splitext(basename(image_name))[0] + '_all_detected' + splitext(image_name)[1]
    detected_file_name = splitext(basename(image_name))[0] + '_detected' + splitext(image_name)[1]

    print "Saving to {}".format(path + '/' + detected_file_name)
    scipy.misc.imsave(path + '/' + true_file_name, true_annotated)
    scipy.misc.imsave(path + '/' + all_detected_file_name, all_detected_annotated)
    scipy.misc.imsave(path + '/' + detected_file_name, detected_annotated)

    with open(path + '/' + splitext(basename(image_name))[0] + '_annotations.txt', 'w') as f:
        for annotation in annotations:
            f.write("True     {0}, {1} - {2}, {3}\n".format(annotation[0], annotation[1], annotation[2], annotation[3]))
        for annotation in minimal:
            f.write("Detected {0}, {1} - {2}, {3}\n".format(annotation[0], annotation[1], annotation[2], annotation[3]))

parser = argparse.ArgumentParser(description='Processes images and detects pedestrians.')
parser.add_argument('--descriptor', choices=['hog', 'css', 'hogcss'], required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()

print "Loading model..."

if args.descriptor == "hog":
    descriptor = HOG()
elif args.descriptor == "css":
    descriptor = CSS()
elif args.descriptor == "hogcss":
    descriptor = HOGCSS()

model = Model(descriptor)
model.load(args.model)

print "Loading dataset..."
images_train, annotations_train = load_full(FULL_TRAIN_IMAGES.format(args.input), FULL_TRAIN_ANNOTATIONS.format(args.input))
images_test, annotations_test = load_full(FULL_TEST_IMAGES.format(args.input), FULL_TEST_ANNOTATIONS.format(args.input))

print "Detecting..."
data = zip(images_train, annotations_train)
if not exists(RESULTS_TRAIN.format(args.output)):
    makedirs(RESULTS_TRAIN.format(args.output))
for i, pair in enumerate(data):
    process(model, RESULTS_TRAIN.format(args.output), pair, i, len(data))

data = zip(images_test, annotations_test)
if not exists(RESULTS_TEST.format(args.output)):
    makedirs(RESULTS_TEST.format(args.output))
for i, pair in enumerate(data):
    process(model, RESULTS_TEST.format(args.output), pair, i, len(data))