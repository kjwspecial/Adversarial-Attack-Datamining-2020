import tensorflow as tf
from tensorflow.keras import models, layers, losses, optimizers, metrics

def diff_x(input, r):
    assert len(input.shape) == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = tf.concat([left, middle, right], axis=2)
    return output

def diff_y(input, r):
    assert len(input.shape) == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = tf.concat([left, middle, right], axis=3)

    return output

class BoxFilter(models.Model):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def call(self, x):
        assert len(x.shape) == 4
        return diff_y(tf.math.cumsum( diff_x(tf.math.cumsum(x, axis=2),self.r), axis=3), self.r)