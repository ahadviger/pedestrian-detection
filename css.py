import numpy as np

from skimage.exposure import histogram
from scipy.ndimage import zoom

def _hist2(cell, bins):
    d = cell.shape[2]
    histograms = []
    for i in range(d):
        histograms.append(histogram(cell[:, :, i], bins)[0])
    return np.hstack(histograms)

def _hist(cell, bins):
    sy, sx, d = cell.shape
    step = 1.0 / bins

    result = np.zeros((bins * bins * bins))
    for y in range(sy):
        for x in range(sx):
            bin_1 = min(int(cell[y, x, 0] / step), bins - 1)
            bin_2 = min(int(cell[y, x, 1] / step), bins - 1)
            bin_3 = min(int(cell[y, x, 2] / step), bins - 1)
            result[bin_1 * bins * bins + bin_2 * bins + bin_3] += 1

    return result

def css(image, bins=10, pixels_per_cell=(8, 8), visualize=False, feature_vector=True):
    sy, sx, d = image.shape
    cx, cy = pixels_per_cell

    n_cellsx = int(np.floor(sx // cx))
    n_cellsy = int(np.floor(sy // cy))

    cells = np.zeros((n_cellsy, n_cellsx, bins ** d))
    for y in range(n_cellsy):
        for x in range(n_cellsx):
            cells[y, x, :] = _hist(image[y*cy:(y+1)*cy, x*cx:(x+1)*cx, :], bins)

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
