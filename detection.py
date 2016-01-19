import cPickle

#import matplotlib.pyplot as plt
import scipy

from load import *
from skimage import data, io, color
from skimage.transform import pyramid_gaussian, pyramid_expand
from skimage.feature import hog
from sklearn.svm import SVC
from PIL import Image, ImageDraw
from os.path import join, splitext, exists, basename

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

def process(model, path, pair):
    start = datetime.now()

    (image_name, image), annotations = pair
    print "Detecting: " + image_name

    detected = model.detect(image)
    end = datetime.now()

    print "Detection time: " + str(end - start)

    true_annotated = annotate(image, annotations)
    detected_annotated = annotate(image, detected)

    true_file_name = splitext(basename(image_name))[0] + '_true' + splitext(image_name)[1]
    detected_file_name = splitext(basename(image_name))[0] + '_detected' + splitext(image_name)[1]

    scipy.misc.imsave(path + '/' + true_file_name, true_annotated)
    scipy.misc.imsave(path + '/' + detected_file_name, detected_annotated)

    with open(path + '/' + splitext(basename(image_name))[0] + '_annotations.txt', 'w') as f:
        for annotation in annotations:
            f.write("True     {0}, {1} - {2}, {3}\n".format(annotation[0], annotation[1], annotation[2], annotation[3]))
        for annotation in detected:
            f.write("Detected {0}, {1} - {2}, {3}\n".format(annotation[0], annotation[1], annotation[2], annotation[3]))

print "Loading model..."
descriptor = HOG()
model = Model(descriptor)
model.load('hog4')

print "Loading dataset..."
images_train, annotations_train = load_full(FULL_TRAIN_IMAGES, FULL_TRAIN_ANNOTATIONS)
images_test, annotations_test = load_full(FULL_TEST_IMAGES, FULL_TEST_ANNOTATIONS)

data = zip(images_train, annotations_train)
print "Detecting..."
for pair in data:
    process(model, RESULTS_TRAIN, pair)