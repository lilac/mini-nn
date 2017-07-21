import struct
import numpy as np
from os import path
import gzip


def read_labels(fpath):
    with open(fpath, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.int8)
        return labels


def read_images(fpath):
    with open(fpath, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        return images


def read(fpath ="../data"):
    train_images_path = path.join(fpath, 'train-images-idx3-ubyte')
    train_labels_path = path.join(fpath, 'train-labels-idx1-ubyte')
    test_images_path = path.join(fpath, 't10k-images-idx3-ubyte')
    test_labels_path = path.join(fpath, 't10k-labels-idx1-ubyte')

    train_images = read_images(train_images_path)
    train_labels = read_labels(train_labels_path)
    assert len(train_images) == len(train_labels)
    test_images = read_images(test_images_path)
    test_labels = read_labels(test_labels_path)
    assert len(test_images) == len(test_labels)

    return (train_images, train_labels), (test_images, test_labels)


def vectorize(a):
    e = np.zeros((10, 1))
    e[a] = 1.0
    return e


def reshape(data, vectorize_y=False):
    images = [np.reshape(x, (784, 1)).astype(np.float32) for x in data[0]]
    # norm the matrix
    images = [image * (1.0 / 255) for image in images]
    labels = [vectorize(y) for y in data[1]] if vectorize_y else data[1]
    return zip(images, labels)


def load():
    import random
    train_data, test_data = read()
    trains = reshape(train_data)
    random.shuffle(trains)
    tests = reshape(test_data)
    samples = [(image, vectorize(y)) for image, y in trains[:50000]]
    return samples, trains[50000:], tests

if __name__ == '__main__':
    train_set, validation_set, test_set = load()
    print "Training {}, test {}".format(len(train_set), len(test_set))
