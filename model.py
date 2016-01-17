import cPickle

import matplotlib.pyplot as plt
import imutils

from constants import *

import scipy
from os import makedirs
from os.path import exists
from load import *
from skimage import data, io, color
from skimage.transform import pyramid_gaussian, pyramid_expand

from sklearn.svm import SVC
from datetime import datetime
from PIL import Image, ImageDraw
from prepare_dataset import *

class Model(object):
    def __init__(self, descriptor, max_layer=12, downscale=1.05,
            window_size=(64, 128), step_size=8, threshold=0.65):
        self.descriptor = descriptor
        self.max_layer = max_layer
        self.downscale = downscale
        self.window_size = window_size
        self.step_size = step_size
        self.threshold = threshold

    def sliding_windows(self, image):
        for y in xrange(0, image.shape[0], self.step_size):
            for x in xrange(0, image.shape[1], self.step_size):
                window = image[y:y + self.window_size[1], x:x + self.window_size[0]]
                if np.shape(window) == (self.window_size[1], self.window_size[0], 3):
                    yield ((x, y), window)

    def scaled_sliding_windows(self, image):
        pyramid = pyramid_gaussian(image, max_layer=self.max_layer, downscale=self.downscale)
        for layer in pyramid:
            scale = float(np.shape(image)[0]) / np.shape(layer)[0]
            windows = self.sliding_windows(layer)
            for (x, y), window in windows:
                yield (window, [x * scale, y * scale, (x + self.window_size[0]) * scale, (y + self.window_size[1]) * scale])

    def descriptor_sliding_windows(self, image):
        for window, position in self.scaled_sliding_windows(image):
            yield (position, window, self.descriptor.extract(window))

    def non_maximum_suppression(self, annotations):
        if len(annotations) == 0:
            return []

        x1 = annotations[:, 0]
        y1 = annotations[:, 1]
        x2 = annotations[:, 2]
        y2 = annotations[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(area)

        result = []

        print(annotations[idxs])
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
            print(overlap)

            idxs = np.delete(idxs,
                np.concatenate(([last], np.where(overlap > self.threshold)[0])))

        return annotations[result]

    def detect(self, image):
        image = imutils.resize(image, width=min(400, image.shape[1]))

        start = datetime.now()

        data = list(self.descriptor_sliding_windows(image))
        end = datetime.now()

        print "Feature extraction time: " + str(end - start)

        start = datetime.now()

        detected = []
        all_features = [features for position, window, features in data]
        predictions = self.model.predict(all_features)
        for (position, window, features), prediction in zip(data, predictions):
            if prediction == 1:
                detected.append(position)

        end = datetime.now()
        print "Feature prediction time: " + str(end - start)

        return self.non_maximum_suppression(self.non_maximum_suppression(np.array(detected)))
    
    def train(self, images_train, labels_train, images_test, labels_test):
        print "Extracting features..."
        descriptor_train = self.descriptor.extract_all(images_train)
        descriptor_test = self.descriptor.extract_all(images_test)
    
        print "Training SVM..."
        self.model = SVC()
        self.model.fit(descriptor_train, labels_train)

        print "Train score:", self.model.score(descriptor_train, labels_train)
        print "Test score:", self.model.score(descriptor_test, labels_test)

    def prepare_hard_negative(self, prefix):
        print "Loading dataset..."
        images_train, labels_train = load(WINDOW_TRAIN_POS, WINDOW_TRAIN_NEG)
        images_test, labels_test = load(WINDOW_TEST_POS, WINDOW_TRAIN_NEG)
        
        print "Initial training..."
        self.train(images_train + images_test, labels_train + labels_test,
                   images_train + images_test, labels_train + labels_test)

        print "Searching for hard negative..."
        counter = 0
        for name, image in load_inria(INRIA_TRAIN_NEG):
            print "Processing: " + name
            for position, window, features in descriptor_sliding_windows(image):
                if self.model.predict([features])[0] == 1:
                    scipy.misc.imsave(WINDOW_TRAIN_NEG + '/' + prefix + str(counter) + '.jpg', window)
                    counter += 1

        counter = 0
        for image in load_inria(INRIA_TEST_NEG):
            print "Processing: " + name
            for position, window, features in descriptor_sliding_windows(image):
                if self.model.predict([features])[0] == 1:
                    scipy.misc.imsave(WINDOW_TEST_NEG + '/' + prefix + str(counter) + '.jpg', window)
                    counter += 1

    def prepare_initial(self):
        for dir in [WINDOW_TRAIN_NEG, WINDOW_TEST_NEG, WINDOW_TRAIN_POS, WINDOW_TEST_POS]:
            makedirs(dir)
    
        make_dataset(INRIA_TRAIN_NEG, WINDOW_TRAIN_NEG, center='random', sample=10)
        make_dataset(INRIA_TEST_NEG, WINDOW_TEST_NEG, center='random', sample=10)

        make_dataset(INRIA_TRAIN_POS, WINDOW_TRAIN_POS,  center='center')
        make_dataset(INRIA_TEST_POS, WINDOW_TEST_POS, center='center')

    def save(self, name):
        with open('data/' + name + '.pkl', 'wb') as fid:
            cPickle.dump(self.model, fid)

    def load(self, name):
        with open('data/' + name + '.pkl', 'rb') as fid:
            self.model = cPickle.load(fid)
