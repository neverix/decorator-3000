from matplotlib import pyplot as plt
import numpy as np
from math import sqrt
from functools import reduce


def main():
    # np.random.seed(33)
    space = make_space()
    net = CombineSubstrate(LinearSubstrate(3), RadialSubstrate(2), VerticalSubstrate(4),  # )
                           PointSubstrate(7))
    results = net(space)
    plot_results(results)


def plot_results(results):
    plt.axis("equal")
    for x, y in results:
        if len(x) == 1:
            plt.scatter(x, y, c='b')
        else:
            plt.plot(x, y, c='b')
    plt.show()


class CombineSubstrate():
    def __init__(self, *substrates):
        self.substrates = substrates

    def __call__(self, input):
        lines = []
        for substrate in self.substrates:
            lines.extend(substrate(input))
        return lines


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


class RadialSubstrate():
    def __init__(self, out_lines=3, baseline=0.2,
                 features=sin_feats([0, 0, 1, 1, 1, 1]), **kwargs):
        self.substrate = LinearSubstrate(features=features, out_lines=out_lines,
                                         **kwargs)
        self.baseline = baseline

    def __call__(self, space):
        lines = self.substrate(space)
        for i, (x, y) in enumerate(lines):
            y = np.abs(y)
            xy = np.stack([np.sin(x), np.cos(x)], axis=0) * (y + self.baseline)
            xy[0] += np.pi
            lines[i] = xy
        return lines


class VerticalSubstrate(object):
    def __init__(self, lines=5, layers=2, features=vert_feats, **kwargs):
        self.lines = lines
        self.substrate = LinearSubstrate(
            out_lines=1, features=features, layers=layers,
            **kwargs)

    def __call__(self, space):
        space = space[np.newaxis, ...]
        space = np.repeat(space, self.lines, axis=0)
        space = self.substrate(space)
        space = space.reshape((-1, *space.shape[2:]))
        space[..., [0, 1], :] = space[..., [1, 0], :]
        space[..., 1, :] = space[..., 1, :] / np.pi - 1
        space[..., 0, :] -= space[..., 0, :].mean(axis=-1)[:, np.newaxis]
        space[..., 0, :] += np.linspace(0,
                                        np.pi * 2, self.lines)[:, np.newaxis]
        return space


class PointSubstrate(object):
    def __init__(self, lines=5, out_lines=1,
                 x_sensitivity=1, y_sensitivity=1,
                 *args, **kwargs):
        self.lines = lines
        self.substrate = LinearSubstrate(*args, out_lines=out_lines,
                                         x_sensitivity=x_sensitivity,
                                         y_sensitivity=y_sensitivity, **kwargs)

    def __call__(self, space):
        space = space[..., np.newaxis, 0:1, :]
        space = np.repeat(space, self.lines, axis=0)
        space[:, :, 0] += np.linspace(0, np.pi * 2,
                                      self.lines)[..., np.newaxis]
        space = self.substrate(space)
        space = space.reshape((-1, *space.shape[2:]))
        return space


def make_space(high=np.pi * 2):
    return np.linspace(0, high)[:, np.newaxis]


class LinearSubstrate(object):
    def __init__(self, out_lines=4, features=sin_feats([1, 1, 1, 1]),
                 hidden=5, layers=4, nn_args=dict(),
                 x_sensitivity=0, y_sensitivity=1):
        self.features = features
        self.net = MLP(features(make_space()).shape[-1], hidden,
                       out_lines * 2, layers, **nn_args)
        self.x_sensitivity = x_sensitivity
        self.y_sensitivity = y_sensitivity

    def __call__(self, space):
        features = self.features(space.copy())
        lines = self.net(features)
        x, y = lines[..., :, ::2], lines[..., :, 1::2]
        x = space + x * self.x_sensitivity
        y_mean = np.mean(y, axis=-2)[..., np.newaxis, :]
        y = y_mean + (y - y_mean) * self.y_sensitivity
        lines = np.stack([x, y], axis=-2).swapaxes(-1, -3)
        return lines


def make_grid(w, h):
    x, y = np.mgrid[0:h, 0:w]
    xy = np.concatenate(
        [x[..., np.newaxis], y[..., np.newaxis]], axis=-1)
    xy = xy.astype(float)
    xy[..., 0] /= h
    xy[..., 1] /= h
    return xy


class MLP(object):
    def __init__(self, input, hidden, output, layers=1,
                 activation_hidden=np.tanh, activation_output=np.tanh):
        self.layers = []
        self.layers.append(Layer(input, hidden, activation=activation_hidden))
        for _ in range(layers):
            self.layers.append(
                Layer(hidden, hidden, activation=activation_hidden))
        self.layers.append(Layer(hidden, output, activation=activation_output))

    def __call__(self, input):
        return reduce(lambda x, l: l(x), self.layers, input)


class Layer(object):
    def __init__(self, input, output, activation=np.tanh):
        self.weight = np.random.normal(
            scale=sqrt(1/input), size=(input + 1, output))
        self.activation = activation

    def __call__(self, input):
        input = np.concatenate(
            [input, np.ones_like(input[..., :1])], axis=-1)  # bias
        return self.activation(input @ self.weight)


if __name__ == "__main__":
    main()
