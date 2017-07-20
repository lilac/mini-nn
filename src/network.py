import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def derivatives(x, y, z):
    """
    Given the loss function C(x, y) = (f(x, w, b) - y)^2
    f(x, w, b) = sigmoid(g(x, w, b))
    g(x, w, b) = wx + b

    the partial derivatives d.C / d.b =  2(f(x, w, b) - y) * (d.f(x, w, b) / d.b)
    and it's known that
      d.sigmoid(x) / d.x = sigmoid(x) * (1 - sigmoid(x))
      d.g(x, w, b) / d.b = 1
      d.g(x, w, b) / d.w = x^T (x transpose)
    so
    d.C / d.b = 2(f(x, w, b) - y) * sigmoid(g(x, w, b)) * (1 - sigmoid(g(x, w, b)) * (d.g(x, w, b) / d.b)
        = 2 (z - y) z (1 - z)
        where z = f(x, w, b)
    d.C / d.w = d.C / d.b * x^T

    :param x:
    :param y:
    :param z: z = f(x, w, b)
    :return: d.C / d.w, d.C / d.b
    """
    bDerivative = (z - y) * z * (1 - z)
    wDerivative = bDerivative * x.transpose()
    return wDerivative, bDerivative


class Network(object):
    def __init__(self, sizes):
        self.nLayers = len(sizes)
        self.sizes = sizes
        self.weight = [np.random.randn(r, c) for c, r in zip(sizes[:-1], sizes[1:])]
        self.bias = [np.random.randn(r, 1) for r in sizes[1:]]

    def feed_forward(self, x):
        for w, b in zip(self.weight, self.bias):
            z = np.dot(w, x) + b
            x = sigmoid(z)
        return x

    def feed(self, x):
        xs = [x]  # input layer by layer
        zs = []  # weighted sum layer by layer
        for w, b in zip(self.weight, self.bias):
            z = np.dot(w, x) + b
            zs.append(z)
            x = sigmoid(z)
            xs.append(x)
        return xs, zs

    def back_propagation(self, x, y):
        wGrad = [np.zeros(w.shape) for w in self.weight]
        bGrad = [np.zeros(b.shape) for b in self.bias]
        xs, zs = self.feed(x)
        wGrad[-1], bGrad[-1] = derivatives(xs[-2], y, xs[-1])
        for l in xrange(2, self.nLayers):
            # Wondering why not the commented line
            # bGrad[-l] = np.dot(bGrad[-l + 1], self.weight[-l + 1]) * (1 - xs[-l]) * xs[-l]
            bGrad[-l] = np.dot(self.weight[-l + 1].transpose(), bGrad[-l + 1]) * xs[-l] * (1 - xs[-l])
            wGrad[-l] = bGrad[-l] * xs[-l - 1].transpose()
        return wGrad, bGrad

    def learn(self, batch, eta):
        wGradSum = [np.zeros(w.shape) for w in self.weight]
        bGradSum = [np.zeros(b.shape) for b in self.bias]
        for x, y in batch:
            wGrad, bGrad = self.back_propagation(x, y)
            wGradSum = [s + w for s, w in zip(wGradSum, wGrad)]
            bGradSum = [s + b for s, b in zip(bGradSum, bGrad)]
        self.weight = [w - eta / len(batch) * delta for w, delta in zip(self.weight, wGradSum)]
        self.bias = [b - eta / len(batch) * delta for b, delta in zip(self.bias, bGradSum)]

    def infer(self, x):
        return np.argmax(self.feed_forward(x))

    def evaluate(self, data):
        results = [(self.infer(x), y) for x, y in data]
        return sum(int(z == y) for z, y in results)

    def train(self, data, epochs, batch_size, eta, validation_data=None):
        import random
        import time
        startTime = time.time()

        for i in xrange(epochs):
            random.shuffle(data)
            batches = [data[j: j + batch_size] for j in xrange(0, len(data), batch_size)]
            for batch in batches:
                self.learn(batch, eta)
            runningTime = time.time() - startTime
            if validation_data:
                num = self.evaluate(validation_data)
                print "{} Epoch {}: {} / {} correct".format(runningTime, i, num, len(validation_data))
            else:
                print "{} Epoch {} complete".format(runningTime, i)
