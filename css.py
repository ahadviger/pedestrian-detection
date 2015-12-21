import numpy as np

from skimage.exposure import histogram
from scipy.ndimage import zoom

def css(image, bins=32, pixels_per_cell=(8, 8), visualize=False, feature_vector=True):
    sy, sx, d = image.shape
    cx, cy = pixels_per_cell

    n_cellsx = int(np.floor(sx // cx))
    n_cellsy = int(np.floor(sy // cy))

    cells = np.zeros((n_cellsy, n_cellsx, d * bins))
    for y in range(n_cellsy):
        for x in range(n_cellsx):
            histograms = []
            for i in range(d):
                cell = image[y*cy:(y+1)*cy, x*cx:(x+1)*cx, i]
                histograms.append(histogram(cell, bins)[0])
            cells[y, x, :] = np.hstack(histograms)

    n_cells = n_cellsy * n_cellsx
    pairs = np.zeros((n_cells, n_cells))
    for i in range(n_cells):
        for j in range(n_cells):
            i_y = int(i / n_cellsx)
            i_x = i % n_cellsx
            j_y = int(j / n_cellsx)
            j_x = j % n_cellsx

            pairs[i, j] = np.linalg.norm(cells[i_y, i_x, :] - cells[j_y, j_x, :])
#            pairs[i, j] = np.minimum(cells[i_y, i_x, :], cells[j_y, j_x, :]).sum()

    normalized = pairs / pairs.max()
    
    if visualize:
        result = []
        for i in range(n_cells):
            result_image = np.zeros((n_cellsy * cy, n_cellsx * cx))
            for k in range(n_cells):
                k_y = int(k / n_cellsx)
                k_x = k % n_cellsx
                for l1 in range(cy):
                    for l2 in range(cx):
                        result_image[k_y * cy + l1, k_x * cx + l2] = normalized[i, k]
#            current = [zoom([[cell]], (cy, cx), order=0) for cell in normalized[i, :]]
#            print(np.array(current).shape)
#            result_image = np.array(current).reshape((n_cellsy * cy, n_cellsx * cx))
#            result_image = normalized[i, :].reshape(n_cellsy, n_cellsx)
            result.append(result_image)
        print(result[0].shape)
        return result
