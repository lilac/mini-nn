import mnist_loader as loader
import mnist
import network

EPOCHS = 100
BATCH_SIZE = 10
ETA = 0.1
SIZES = [28 * 28, 15, 10]

# (train_set, validation_set, test_set) = loader.load_data_wrapper()
(train_set, validation_set, test_set) = mnist.load()


def train(args):
    import cPickle
    import gzip

    net = network.Network(args.sizes)
    if args.validate:
        net.train(train_set, args.epochs, args.batch, args.eta, validation_data=validation_set)
    else:
        net.train(train_set, args.epochs, args.batch, args.eta)
    print "{}".format(args)
    evaluate_data(net, test_set)
    # save model
    f = gzip.open(args.output, "w")
    cPickle.dump(net, f)


def evaluate(args):
    import gzip
    import cPickle
    f = gzip.open(args.model, "rb")
    net = cPickle.load(f)
    print "Network layers: {}".format(net.sizes)
    if args.dir:
        infer(net, args.dir)
    else:
        evaluate_data(net, test_set)


def rgb2gray(rgb):
    import numpy as np

    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def infer(net, fpath):
    from scipy import misc, ndimage
    import numpy as np
    import glob

    for image_path in glob.glob(fpath + "/*/*.png"):
        # image = ndimage.imread(image_path, mode='RGB')
        # image = rgb2gray(image)
        image = ndimage.imread(image_path, flatten=True)
        img = misc.imresize(image, (28, 28))
        # image.resize((28, 28, 1))
        x = img.reshape((784, 1)).astype(np.float32)
        x *= 1.0 / 255
        res = net.infer(x)
        print "{}: infer as {}".format(image_path, res)


def evaluate_data(net, data):
    num = net.evaluate(data)
    print "Test error rate: {}%".format(100 - 100.0 * num / len(data))


def gen_file_name():
    import time
    from os import path

    timestr = time.strftime("%Y%m%d-%H%M%S")
    return path.join("../models", timestr + ".pkl.gzip")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    trainParser = subparsers.add_parser("t", help="train model")
    evalParser = subparsers.add_parser("e", help="evaluate model")
    # parser.add_argument("t", "train", help="train model", action="store_true")
    # parser.add_argument("e", "evaluate", action="store_true", help="evaluate model")
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    trainParser.add_argument("-n", "--epochs", type=int, default=EPOCHS)
    trainParser.add_argument("-b", "--batch", type=int, default=BATCH_SIZE)
    trainParser.add_argument("-e", "--eta", type=float, default=ETA)
    trainParser.add_argument("-l", "--sizes", metavar='N', type=int, nargs='+',
                             help='sizes of the layers', default=SIZES)
    trainParser.add_argument("--validate", action="store_true", default=False)

    trainParser.add_argument("-o", "--output", default=gen_file_name())
    trainParser.set_defaults(func=train)

    evalParser.add_argument("model", help="model file")
    evalParser.add_argument('-d', "--dir", help="test images directory")
    evalParser.set_defaults(func=evaluate)

    args = parser.parse_args()
    print "Args: {}".format(args)
    args.func(args)


if __name__ == '__main__':
    main()
