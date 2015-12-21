from css import *

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from skimage import data, color

image = data.imread("crop001061.png")

#plt.imshow(image)
#plt.show()

image_hsv = color.rgb2hsv(image)

visualize = [13 * 34 + 14, 25 * 34 + 17, 39 * 34 + 16, 47 * 34 + 28]
css_images = css(image_hsv, pixels_per_cell=(8, 8), visualize = visualize)

print(len(css_images))

for i in range(len(css_images)):
    print(i)
    plt.imshow(css_images[i], cmap = cm.Greys_r)
    plt.show()
