import cPickle

import matplotlib.pyplot as plt

from constants import *

from random import shuffle
import scipy
from os import makedirs
from os.path import join, splitext, exists, basename
from load import *
from skimage import data, io, color
from skimage.transform import pyramid_gaussian, pyramid_expand

from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix
from datetime import datetime
from PIL import Image, ImageDraw
from prepare_dataset import *

class Model(object):
    def __init__(self, descriptor, max_layer=-1, downscale=1.1,
            window_size=(64, 128), step_size=12, threshold=0.3):
        self.descriptor = descriptor
        self.max_layer = max_layer
        self.downscale = downscale
        self.window_size = window_size
        self.step_size = step_size
        self.threshold = threshold

    def sliding_windows(self, width, height):
        for y in xrange(0, height - self.window_size[1] + 1, self.step_size):
            for x in xrange(0, width - self.window_size[0] + 1, self.step_size):
                yield (x, x + self.window_size[0], y, y + self.window_size[1])

    def pyramid(self, image, min_size=(64, 128), downscale=1.1):
        scale = downscale
        height, width, _ = np.shape(image)
        current_height, current_width = height, width

        while current_height > min_size[1] and current_width > min_size[0]:
            current_height = int(height / scale)
            current_width = int(width / scale)
            scale = scale * downscale
            yield (current_width, current_height, scale)

    def scaled_sliding_windows(self, image, dimensions):
        counter = 0
        im = Image.fromarray(image)

        for (width, height, scale) in self.pyramid(image, downscale=self.downscale):
            scale_ = float(dimensions[0]) / height
            windows = self.sliding_windows(width, height)
            resized = im.resize((width, height), Image.ANTIALIAS)
            for (x1, x2, y1, y2) in windows:
                counter += 1
                cropped = resized.crop((x1, y1, x2, y2))
#                cropped.save('data/tmp/' + str(counter) + 'a.jpg', format='JPEG', subsampling=0, quality=100)               
                yield (np.array(cropped), [int(x1 * scale_), int(y1 * scale_), int(x2 * scale_), int(y2 * scale_)])

    def descriptor_sliding_windows(self, image, dimensions):
        windows = list(self.scaled_sliding_windows(image, dimensions))
        all_features = self.descriptor.extract_all([window for window, position in windows])

        for (window, position), features in zip(windows, all_features):
            yield (position, window, features)

    def non_maximum_suppression(self, annotations):
        if len(annotations) == 0:
            return []

        x1 = annotations[:, 0]
        y1 = annotations[:, 1]
        x2 = annotations[:, 2]
        y2 = annotations[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        area = area.astype(float)
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
            overlap = w * h / area[idxs[:last]]#np.minimum(area[i], area[idxs[:last]])

            idxs = np.delete(idxs,
                np.concatenate(([last], np.where(overlap > self.threshold)[0])))

        return annotations[result]

    def detect(self, image):
        dimensions = np.shape(image)
        im = Image.fromarray(image)
        new_width = min(400, image.shape[1])
        new_height = new_width * image.shape[0] / image.shape[1]
        im = im.resize((new_width, new_height), Image.ANTIALIAS)
        image = np.array(im)

        start = datetime.now()

        data = list(self.descriptor_sliding_windows(image, dimensions))
        end = datetime.now()

        print "Feature extraction time: " + str(end - start)

        start = datetime.now()

        detected = []
        all_features = [features for position, window, features in data]
        predictions = self.model.predict(all_features)
        for (position, window, features), prediction in zip(data, predictions):
            if prediction == 1:
#                plt.imshow(window)
#                plt.show()
                detected.append(position)

        end = datetime.now()
        print "Feature prediction time: " + str(end - start)

        return np.array(detected)
    
    def train(self, images_train, labels_train, images_test, labels_test):
        start = datetime.now()
        print "Extracting features..."
        descriptor_train = self.descriptor.extract_all(images_train)
        descriptor_test = self.descriptor.extract_all(images_test)
        end = datetime.now()
        print "Feature extraction time: " + str(end - start)
    
        start = datetime.now()
        print "Training SVM..."
        self.model = LinearSVC()
        self.model.fit(descriptor_train, labels_train)
        end = datetime.now()
        print "SVM training time: " + str(end - start)

        start = datetime.now()
        print "Train score:", self.model.score(descriptor_train, labels_train)
        print confusion_matrix(labels_train, self.model.predict(descriptor_train))
        print "Test score:", self.model.score(descriptor_test, labels_test)
        print confusion_matrix(labels_test, self.model.predict(descriptor_test))
        end = datetime.now()
        print "SVM evaluation time: " + str(end - start)

    def prepare_hard_negative(self, prefix):
        print "Searching for hard negative..."

        def process(inria_path, window_path):
            images = load_inria(inria_path)
            for i, (name, image) in enumerate(images):
                print "Processing [{0}/{1}]: {2}".format(i, len(images), name)
                data = list(self.descriptor_sliding_windows(image, np.shape(image)))
                all_features = [features for position, window, features in data]
                predictions = self.model.predict(all_features)

                hard = []
                for (position, window, features), prediction in zip(data, predictions):
                    if prediction == 1:
                        hard.append(window)
                shuffle(hard)
                counter = 0
                for window in hard:
                    if counter < 3:
                        new_file_name = window_path + '/' + prefix + splitext(name)[0] + '_' + str(counter) + '.jpg'
                        scipy.misc.imsave(new_file_name, window)
                        print(new_file_name)
                        counter += 1

        process(INRIA_TRAIN_NEG, WINDOW_TRAIN_NEG)
        process(INRIA_TEST_NEG, WINDOW_TEST_NEG)

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
