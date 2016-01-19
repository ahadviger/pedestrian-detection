import numpy as np

from skimage import data, io, color
from skimage.exposure import histogram
from skimage.feature import hog
from scipy.ndimage import zoom
from scipy.spatial.distance import cdist
from datetime import datetime
from joblib import Parallel, delayed

def helper(descriptor, i, n, image, visualize, feature_vector):
    if i > 0 and i % 1000 == 0:
        print "Descriptor progress {0}/{1}".format(i, n)
    return descriptor.extract(image, visualize, feature_vector)

class ImageDescriptor(object):
    def feature_vector_size(self, image):
        return self.extract(image).shape[0]

    def extract(self, image, visualize=False, feature_vector=True):
        raise NotImplementedError("Not implemented")

    def extract_all(self, images, visualize=False, feature_vector=True):
        return Parallel(n_jobs=4)(delayed(helper)(self, i, len(images), image, visualize, feature_vector) for i, image in enumerate(images))
#        result = np.empty((len(images), self.feature_vector_size(images[0])))

#        counter = 0
#        start = datetime.now()
#        for i, image in enumerate(images):
#            result[i, :] = self.extract(image, visualize=visualize, feature_vector=feature_vector)

#            counter += 1
#            if counter % 1000 == 0:
#                end = datetime.now()
#                print("{0}/{1}: {2}".format(counter, len(images), end - start))
#                start = end

#        return result

class HOG(ImageDescriptor):
    def __init__(self, bins = 9, pixels_per_cell = (8, 8), cells_per_block = (1, 1)):
        self.bins = bins
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def extract(self, image, visualize=False, feature_vector=True):
        return hog(color.rgb2gray(image), orientations=self.bins,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block, visualise=visualize)

class CSS(ImageDescriptor):
    def __init__(self, bins=9, pixels_per_cell=(8, 8)):
        self.bins = bins
        self.pixels_per_cell = pixels_per_cell

    def extract(self, image, visualize=False, feature_vector=True):
        image = color.rgb2hsv(image)
        sy, sx, d = image.shape
        cx, cy = self.pixels_per_cell

        n_cellsx = int(np.floor(sx // cx))
        n_cellsy = int(np.floor(sy // cy))

        cells = np.zeros((n_cellsy, n_cellsx, self.bins ** d))
        step = 1.0 / self.bins
        bins = np.minimum((image / step).astype(int), self.bins - 1)
        result = bins[:, :, 0] * self.bins * self.bins + bins[:, :, 1] * self.bins + bins[:, :, 2]
        for y in range(n_cellsy):
            for x in range(n_cellsx):
                fx, tx, fy, ty = x*cx, (x+1)*cx, y*cy, (y+1)*cy
                cells[y, x, :] = np.bincount(result[fy:ty, fx:tx].ravel(), minlength=self.bins * self.bins * self.bins)

        n_cells = n_cellsy * n_cellsx
        pairs = np.zeros((n_cells, n_cells))
        cells = cells.reshape((n_cells, self.bins * self.bins * self.bins))
        
        for i in range(n_cells):
            pairs[i, :] = np.minimum(cells[i, :], cells[:, :]).sum(axis=1)
        np.fill_diagonal(pairs, 0)

        m = pairs.max()
        if m > 0:
            normalized = pairs / m
        else:
            print "shit" + str(pairs.min())
            normalized = pairs
        
        if np.count_nonzero(np.isnan(normalized)):
            print np.count_nonzero(np.isnan(normalized))

        if hasattr(visualize, "__len__"):
            indices = visualize
        elif visualize:
            indices = range(n_cells)
        else:
            indices = []

        if len(indices) > 0:
            result = []
            for i in indices:
                result_image = normalized[i, :].reshape((n_cellsy, n_cellsx))
                result.append(result_image)
            return result
        else:
            return np.ravel(normalized)


class HOGCSS(ImageDescriptor):
    def __init__(self, bins = 9, pixels_per_cell = (8, 8), cells_per_block = (1, 1)):
        self.hog = HOG(bins, pixels_per_cell, cells_per_block)
        self.css = CSS(bins, pixels_per_cell)

    def extract(self, image, visualize=False, feature_vector=True):
        return np.concatenate((self.hog.extract(image), self.css.extract(image)))