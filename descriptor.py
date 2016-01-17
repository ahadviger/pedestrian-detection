import numpy as np

from skimage import data, io, color
from skimage.exposure import histogram
from skimage.feature import hog
from scipy.ndimage import zoom

class ImageDescriptor(object):
    def extract(self, image, visualize=False, feature_vector=True):
        raise NotImplementedError("Not implemented")

    def extract_all(self, images, visualize=False, feature_vector=True):
        descriptors = []
        for image in images:
            descriptors.append(self.extract(image, visualize=visualize, feature_vector=feature_vector))
        return descriptors

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
        sy, sx, d = image.shape
        cx, cy = self.pixels_per_cell

        n_cellsx = int(np.floor(sx // cx))
        n_cellsy = int(np.floor(sy // cy))

        cells = np.zeros((n_cellsy, n_cellsx, bins ** d))
        for y in range(n_cellsy):
            for x in range(n_cellsx):
                cells[y, x, :] = self._hist(image[y*cy:(y+1)*cy, x*cx:(x+1)*cx, :])

        n_cells = n_cellsy * n_cellsx
        pairs = np.zeros((n_cells, n_cells))
        for i in range(n_cells):
            for j in range(n_cells):
                i_y = int(i / n_cellsx)
                i_x = i % n_cellsx
                j_y = int(j / n_cellsx)
                j_x = j % n_cellsx

                if i == j:
                    pairs[i, j] = 0
                else:
    #                pairs[i, j] = np.linalg.norm(cells[i_y, i_x, :] - cells[j_y, j_x, :])
                    pairs[i, j] = np.minimum(cells[i_y, i_x, :], cells[j_y, j_x, :]).sum()

        normalized = pairs / pairs.max()
        
        if hasattr(visualize, "__len__"):
            indices = visualize
        elif visualize:
            indices = range(n_cells)
        else:
            indices = []

        if len(indices) > 0:
            result = []
            for i in indices:
    #            result_image = np.zeros((n_cellsy * cy, n_cellsx * cx))
    #            for k in range(n_cells):
    #                k_y = int(k / n_cellsx)
    #                k_x = k % n_cellsx
    #                for l1 in range(cy):
    #                    for l2 in range(cx):
    #                        result_image[k_y * cy + l1, k_x * cx + l2] = normalized[i, k]
                result_image = normalized[i, :].reshape((n_cellsy, n_cellsx))
                result.append(result_image)
            print(result[0].shape)
            return result

    def _hist2(self, cell):
        d = cell.shape[2]
        histograms = []
        for i in range(d):
            histograms.append(histogram(cell[:, :, i], self.bins)[0])
        return np.hstack(histograms)

    def _hist(self, cell):
        sy, sx, d = cell.shape
        step = 1.0 / bins

        result = np.zeros((bins * bins * bins))
        for y in range(sy):
            for x in range(sx):
                bin_1 = min(int(cell[y, x, 0] / step), self.bins - 1)
                bin_2 = min(int(cell[y, x, 1] / step), self.bins - 1)
                bin_3 = min(int(cell[y, x, 2] / step), self.bins - 1)
                result[bin_1 * bins * bins + bin_2 * bins + bin_3] += 1

        return result

class HOGCSS(ImageDescriptor):
    def __init__(self, bins = 9, pixels_per_cell = (8, 8), cells_per_block = (1, 1)):
        self.hog = HOG(bins, pixels_per_cell, cells_per_block)
        self.css = CSS(bins, pixels_per_cell)

    def extract(self, image, visualize=False, feature_vector=True):
        return np.concatenate((self.hog.extract(image), self.css.extract(image)))