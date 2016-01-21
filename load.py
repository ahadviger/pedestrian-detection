import numpy as np
import matplotlib.pyplot as plt
import re
import gc

from os import listdir
from os.path import join, splitext, exists
from skimage import data, io

def is_image(filename):
    ext = [".jpg", ".png", ".gif", "bmp"]
    return any([filename.endswith(e) for e in ext])

def load(pos_dir, neg_dir, hard_examples=[]):
    gc.disable()
    images, labels = [], []
    files = listdir(pos_dir)
    for i, f in enumerate(files):
        if not is_image(f):
            continue
        if "hard_example" in f and (hard_examples is None or not any ([f.startswith(prefix) for prefix in hard_examples])):
            continue
        if i > 0 and i % 1000 == 0:
            print "loaded {0}/{1} positive".format(i, len(files))
        images.append(data.imread(join(pos_dir, f))[:, :, :3])
        labels.append(1)
    files = listdir(neg_dir)
    for i, f in enumerate(files):
        if not is_image(f):
            continue
        if "hard_example" in f and (hard_examples is None or not any ([f.startswith(prefix) for prefix in hard_examples])):
            continue
        if i > 0 and i % 1000 == 0:
            print "loaded {0}/{1} negative".format(i, len(files))
        images.append(data.imread(join(neg_dir, f))[:, :, :3])
        labels.append(0)
    gc.enable()
    return images, labels

def load_inria(image_dir):
    images = []
    image_files = sorted(listdir(image_dir))

    for image_file in image_files:
        if not is_image(image_file):
            continue
        image_path = join(image_dir, image_file)
        images.append((image_file, data.imread(image_path)[:, :, :3]))

    return images

def load_full(image_dir, annotations_dir):
    images, annotations = [], []
    image_files = sorted(listdir(image_dir))
    annotation_files = sorted(listdir(annotations_dir))

    for image_file in image_files:
        annotation_file = join(annotations_dir, splitext(image_file)[0] + '.txt')
        if not exists(annotation_file) or not is_image(image_file):
            continue
        image_path = join(image_dir, image_file)
        images.append((image_path, data.imread(image_path)[:, :, :3]))
        annotations.append(parse_annotations(annotation_file))

    return images, annotations

def parse_annotations(path):
    result = []
    with open(path) as f:
        for line in f:
            m = re.match(r"Bounding box for [\"\w\d\s\(\),-]+: \((\d+),\s*(\d+)\)\s*-\s*\((\d+),\s*(\d+)\)", line)
            if m is not None:
                result.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))))
    return result
