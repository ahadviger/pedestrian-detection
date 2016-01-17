import cPickle

import matplotlib.pyplot as plt
import imutils
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
model.load('hog')

print "Loading dataset..."
images_train, annotations_train = load_full(FULL_TRAIN_IMAGES, FULL_TRAIN_ANNOTATIONS)
images_test, annotations_test = load_full(FULL_TEST_IMAGES, FULL_TEST_ANNOTATIONS)

data = zip(images_train, annotations_train)
print "Detecting..."
for pair in data:
    process(model, RESULTS_TRAIN, pair)

if False:
    def sliding_windows(image, window_size, step_size):
        result = []
        for y in xrange(0, image.shape[0], step_size):
            for x in xrange(0, image.shape[1], step_size):
                window = image[y:y + window_size[1], x:x + window_size[0]]
                if np.shape(window) == (window_size[1], window_size[0], 3):
                    result.append(((x, y), window))

        return result

    def non_maximum_suppression(annotations, threshold=0.65):
        if len(annotations) == 0:
            return []

        x1 = annotations[:, 0]
        y1 = annotations[:, 1]
        x2 = annotations[:, 2]
        y2 = annotations[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        result = []

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            result.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = w * h / area[idxs[:last]].astype(float)

            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > threshold)[0])))

        return annotations[result]

    def detect(image, model, pyramid_max_layer=-1, downscale=1.05, upscale=1.05, upsample=5, window_size=(64, 128), step_size=8):
        print(image.shape)
        image = imutils.resize(image, width=min(400, image.shape[1]))
        print(image.shape)

        detected = []
        pyramid = pyramid_gaussian(image, max_layer=pyramid_max_layer, downscale=downscale)

    #    pyramid = [image]
    #    current = image
    #    for i in range(upsample):
    #        current2 = pyramid_expand(current, upscale=upscale)
    #        current = current2
    #        pyramid.append(current2)

        for layer in pyramid:
            scale = float(np.shape(image)[0]) / np.shape(layer)[0]
            windows = sliding_windows(layer, window_size, step_size)
            print("layer: " + str(len(windows)))
            for (x, y), window in windows:
                features = hog(color.rgb2gray(window), orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualise=False)
                prediction = model.predict([features])[0]
                #prediction = 0
                if prediction == 1:
                    detected.append([x * scale, y * scale, (x + window_size[0]) * scale, (y + window_size[1]) * scale])

        return np.array(detected)

    def annotate(image, annotations):
        pil = Image.fromarray(image)
        draw = ImageDraw.Draw(pil)

        for [x1, y1, x2, y2] in annotations:
            draw.rectangle([(x1, y1), (x2, y2)])

        return np.asarray(pil)

    train_images_dir, train_annotations_dir = "full/train/pos", "full/train/annotations"
    test_images_dir, test_annotations_dir = "full/test/pos", "full/test/annotations"

    print "Loading dataset..."
    images_train, annotations_train = load_full(train_images_dir, train_annotations_dir)
    images_test, annotations_test = load_full(test_images_dir, test_annotations_dir)

    print "Loading model..."
    with open('classifier.pkl', 'rb') as fid:
        model = cPickle.load(fid)

    print "Detecting..."
    for image, annotations in zip(images_train + images_test, annotations_train + annotations_test):
        detected = detect(image, model)
        annotated = annotate(image, detected)

        plt.imshow(annotated)
        plt.show()

        plt.imshow(annotate(image, non_maximum_suppression(detected)))
        plt.show()

