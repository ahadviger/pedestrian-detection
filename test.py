from css import *

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from skimage import data, color

image = data.imread("crop001061.png")
image_hsv = color.rgb2hsv(image)
css_images = css(image_hsv, pixels_per_cell=(16, 16), visualize = True)

print(len(css_images))

indices = [245, 416]
for i in indices:
    plt.imshow(css_images[i], cmap = cm.Greys)
    plt.show()
