from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt
from functools import reduce
import pickle


def main():
    # np.random.seed(33)
    space = make_space()
    net = MLP(6, 6, 6, 6)
    sub = CombineSubstrate(net, [LinearSubstrate] * 4 + [RadialSubstrate] * 3 +
                           [VerticalSubstrate] * 3 + [PointSubstrate] * 3)
    for _ in range(5):
        results = sub.mutation(0.2)(space)
        plot_results(results)
    results = [list(zip(x, y)) for x, y in results]
    pickle.dump(results, open('lines.pkl', 'wb'))


def plot_results(results):
    plt.axis("equal")
    for x, y in results:
        if len(x) == 1:
            plt.scatter(x, y, c='b')
        else:
            plt.plot(x, y, c='b')
    plt.show()


class Genome(object):
    def copy(self):
        return deepcopy(self)

    def mutate(self, weight):
        pass

    def mutation(self, weight):
        copy = self.copy()
        copy.mutate(weight)
        return copy


class CombineSubstrate(Genome):
    def __init__(self, net, substrates, pick=4):
        self.net = net
        self.substrates = [substrate(net) for substrate in substrates]
        self.weights = np.random.uniform(size=len(substrates))
        self.pick = pick
        self.norm()

    def norm(self):
        self.weights /= self.weights.sum()
        kept = self.weights > sorted(self.weights)[-self.pick]
        self.picked = [s for s, k in zip(self.substrates, kept) if k]

    def __call__(self, input):
        lines = []
        for substrate in self.picked:
            lines.extend(substrate(self.net, input))
        return lines

    def mutate(self, weight):
        for substrate in self.substrates:
            substrate.mutate(weight)
        self.weights += np.random.normal(scale=scale(
            len(self.weights)) * weight, size=self.weights.shape)
        self.norm()


def sin_feats(features):
    def extract(space):
        return np.concatenate([
            np.sin(space * f) * w
            for w, f in zip(features, range(1, len(features) + 1))],
            axis=-1)
    return extract


def vert_feats(x):
    x -= np.mean(x)
    x = np.concatenate([x, np.sin(x * 2), x ** 3, x ** 2], axis=-1)
    return x


class RadialSubstrate(Genome):
    def __init__(self, net, baseline=0.2,
                 features=sin_feats([0, 0, 1, 1, 1, 1]), **kwargs):
        self.substrate = LinearSubstrate(features=features,
                                         net=net, **kwargs)
        self.baseline = baseline

    def __call__(self, net, space):
        lines = self.substrate(net, space)
        for i, (x, y) in enumerate(lines):
            y = np.abs(y)
            xy = np.stack([np.sin(x), np.cos(x)], axis=0) * (y + self.baseline)
            xy[0] += np.pi
            lines[i] = xy
        return lines

    def mutate(self, weight):
        self.substrate.mutate(weight)


class VerticalSubstrate(Genome):
    def __init__(self, net, lines=5, features=vert_feats, **kwargs):
        self.lines = lines
        self.substrate = LinearSubstrate(
            features=features, net=net,
            **kwargs)

    def __call__(self, net, space):
        space = space[np.newaxis, ...]
        space = np.repeat(space, self.lines, axis=0)
        space = self.substrate(net, space)
        space = space.reshape((-1, *space.shape[2:]))
        space[..., [0, 1], :] = space[..., [1, 0], :]
        space[..., 1, :] = space[..., 1, :] / np.pi - 1
        space[..., 0, :] -= space[..., 0, :].mean(axis=-1)[:, np.newaxis]
        space[..., 0, :] += np.linspace(0,
                                        np.pi * 2, self.lines)[:, np.newaxis]
        return space

    def mutate(self, weight):
        self.substrate.mutate(weight)


class PointSubstrate(Genome):
    def __init__(self, net, lines=7,
                 x_sensitivity=1, y_sensitivity=1,
                 *args, **kwargs):
        self.lines = lines
        self.substrate = LinearSubstrate(*args, net=net,
                                         x_sensitivity=x_sensitivity,
                                         y_sensitivity=y_sensitivity, **kwargs)

    def __call__(self, net, space):
        space = space[..., np.newaxis, 0:1, :]
        space = np.repeat(space, self.lines, axis=0)
        space[:, :, 0] += np.linspace(0, np.pi * 2,
                                      self.lines)[..., np.newaxis]
        space = self.substrate(net, space)
        space = space.reshape((-1, *space.shape[2:]))
        return space

    def mutate(self, weight):
        self.substrate.mutate(weight)


def make_space(high=np.pi * 2):
    return np.linspace(0, high)[:, np.newaxis]


class LinearSubstrate(Genome):
    def __init__(self, net, features=sin_feats([0, 1, 1, 1]),
                 x_sensitivity=0, y_sensitivity=1):
        self.features = features
        self.n_features = features(make_space()).shape[-1]

        self.prenet = Layer(self.n_features, net.input)
        self.postnet = Layer(net.output, 2)

        self.x_sensitivity = x_sensitivity
        self.y_sensitivity = y_sensitivity

    def __call__(self, net, space):
        features = self.features(space.copy())
        lines = self.postnet(net(self.prenet(features)))
        x, y = lines[..., 0], lines[..., 1]
        x, y = x[..., np.newaxis], y[..., np.newaxis]
        x = space + x * self.x_sensitivity
        y_mean = np.mean(y, keepdims=False)
        y = y_mean + (y - y_mean) * self.y_sensitivity
        lines = np.stack([x, y], axis=-2).swapaxes(-1, -3)
        return lines

    def mutate(self, weight):
        self.prenet.mutate(weight)
        self.postnet.mutate(weight)


def make_grid(w, h):
    x, y = np.mgrid[0:h, 0:w]
    xy = np.concatenate(
        [x[..., np.newaxis], y[..., np.newaxis]], axis=-1)
    xy = xy.astype(float)
    xy[..., 0] /= h
    xy[..., 1] /= h
    return xy


class MLP(Genome):
    def __init__(self, input, hidden, output, layers=1,
                 activation_hidden=np.tanh, activation_output=np.tanh):
        self.input = input
        self.output = output
        self.layers = []
        self.layers.append(Layer(input, hidden, activation=activation_hidden))
        for _ in range(layers):
            self.layers.append(
                Layer(hidden, hidden, activation=activation_hidden))
        self.layers.append(Layer(hidden, output, activation=activation_output))

    def __call__(self, input):
        return reduce(lambda x, l: l(x), self.layers, input)

    def mutate(self, weight):
        for layer in self.layers:
            layer.mutate(weight)


def scale(n):
    return sqrt(1 / n)


class Layer(Genome):
    def __init__(self, input, output, activation=np.tanh):
        self.input = input
        self.output = output
        self.weight = np.random.normal(
            scale=scale(self.input), size=(input + 1, output))
        self.activation = activation

    def __call__(self, input):
        input = np.concatenate(
            [input, np.ones_like(input[..., :1])], axis=-1)  # bias
        return self.activation(input @ self.weight)

    def mutate(self, weight):
        self.weight += np.random.normal(scale=scale(self.input),
                                        size=self.weight.shape)


if __name__ == "__main__":
    main()
