import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import join
from skimage import data, io
from random import randint

def crop(image, width=64, height=128, center='center'):
    h, w, _ = image.shape
    if center == 'center':
        x = h / 2 - height / 2
        y = w / 2 - width / 2
    elif center == 'random':
        x = randint(0, h - height)
        y = randint(0, w - width)
    return image[x : x + height, y : y + width]

def is_image(filename):
    ext = [".jpg", ".png", ".gif"]
    return any([filename.endswith(e) for e in ext])
    
def make_dataset(input_dir, output_dir, center='center'):
    files = listdir(input_dir)
    for f in files:
        if not is_image(f):
            continue
        image = data.imread(join(input_dir, f))
        io.imsave(join(output_dir, f), crop(image))
        print "Saved " + f
        
make_dataset("C:/Users/Antea/Downloads/INRIAPerson/train_64x128_H96/neg", "train/neg", center='random')
make_dataset("C:/Users/Antea/Downloads/INRIAPerson/test_64x128_H96/neg", "test/neg", center='random')

make_dataset("C:/Users/Antea/Downloads/INRIAPerson/train_64x128_H96/pos", "train/pos", center='center')
make_dataset("C:/Users/Antea/Downloads/INRIAPerson/test_64x128_H96/pos", "test/pos", center='center')