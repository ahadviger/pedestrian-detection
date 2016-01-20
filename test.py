from descriptor import *

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime
from skimage import data, color

image = data.imread("/home/mfolnovic/Downloads/tmp/crop001061.png")

#plt.imshow(image)
#plt.show()

#image_hsv = color.rgb2hsv(image)

css = CSS()
visualize = False#[13 * 34 + 14, 25 * 34 + 17, 39 * 34 + 16, 47 * 34 + 28]
start = datetime.now()
css_images = css.extract(image, visualize = visualize)
end = datetime.now()
print "Time: " + str(end-start)

print(len(css_images))

#for i in range(len(css_images)):
#    print(i)
#    plt.imshow(css_images[i], cmap = cm.Greys_r)
#    plt.show()
