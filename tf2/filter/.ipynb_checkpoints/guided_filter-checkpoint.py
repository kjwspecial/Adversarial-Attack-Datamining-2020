import tensorflow as tf
from tensorflow.keras import models, layers, losses, optimizers, metrics
from filter.box_filter import BoxFilter
#Deep guided input shape : [batch, channel,w, h]
class FastGuidedFilter(models.Model):
    def __init__(self, r, eps=1e-8):
        super(FastGuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def call(self, lr_x, lr_y, hr_x):
        # batch, channel W, H
        lr_x = tf.transpose(lr_x, perm=[0,3,1,2])
        lr_y = tf.transpose(lr_y, perm=[0,3,1,2])
        hr_x = tf.transpose(hr_x, perm=[0,3,1,2])

        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.shape
        n_lry, c_lry, h_lry, w_lry = lr_y.shape
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.shape

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1
        ## N

        N = self.boxfilter(tf.Variable(tf.ones((1, 1, h_lrx, w_lrx))))
    
        ## mean_x
        mean_x = self.boxfilter(lr_x) / N
        ## mean_y
        mean_y = self.boxfilter(lr_y) / N
        ## cov_xy
        cov_xy = self.boxfilter(lr_x * lr_y) / N - mean_x * mean_y
        ## var_x
        var_x = self.boxfilter(lr_x * lr_x) / N - mean_x * mean_x
        ## A
        A = cov_xy / (var_x + self.eps)
        ## b
        b = mean_y - A * mean_x
        A = tf.transpose(A, perm=[0,2,3,1])
        b = tf.transpose(b, perm=[0,2,3,1])
        hr_x = tf.transpose(hr_x, perm=[0,2,3,1])
        mean_A = tf.image.resize(A, [160, 160])
        mean_b = tf.image.resize(b, [160, 160])
        #shape [1, 160, 160, 3]
        return mean_A*hr_x+mean_b