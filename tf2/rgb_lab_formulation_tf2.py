import cv2
import torch
import numpy as np
from scipy import misc
from PIL import Image
import tensorflow as tf
def preprocess_lab(lab):
    L_chan, a_chan, b_chan = lab[:,:,0], lab[:,:,1] ,lab[:,:,2]
    # L_chan: 입력 범위 [0, 100]인 흑백
    # 입력 범위가 [-110, 110] 컬러 채널, 정확하진 않다고함.
    # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
    return [L_chan / 50.0 - 1.0, a_chan / 110.0, b_chan / 110.0]


def deprocess_lab(L_chan, a_chan, b_chan):
    return tf.stack([(L_chan + 1) / 2.0 * 100.0, a_chan * 110.0, b_chan * 110.0], axis=2)

def rgb_to_lab(srgb):

    srgb_pixels = tf.reshape(srgb,[-1,3])
    linear_mask = tf.cast((srgb_pixels <= 0.04045),float)
    exponential_mask = tf.cast ((srgb_pixels > 0.04045),float)
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
    rgb_to_xyz = tf.constant([
                    #    X        Y          Z
                    [0.412453, 0.212671, 0.019334], # R
                    [0.357580, 0.715160, 0.119193], # G
                    [0.180423, 0.072169, 0.950227], # B
                ],float)
    xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)
    xyz_normalized_pixels = tf.cast(tf.math.multiply(xyz_pixels, tf.constant([1/0.950456, 1.0, 1/1.088754])),float)
    epsilon = 6.0/29.0
    linear_mask = tf.cast((xyz_normalized_pixels <= (epsilon**3)),float)

    exponential_mask = tf.cast((xyz_normalized_pixels > (epsilon**3)),float)
    fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4.0/29.0) * linear_mask + ((xyz_normalized_pixels+0.000001) ** (1.0/3.0)) * exponential_mask
    fxfyfz_to_lab = tf.constant([
            #  l       a       b
            [  0.0,  500.0,    0.0], # fx
            [116.0, -500.0,  200.0], # fy
            [  0.0,    0.0, -200.0], # fz
        ],float)
    lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0],float)
    return tf.reshape(lab_pixels, srgb.shape)

def lab_to_rgb(lab):
    lab_pixels = tf.reshape(lab,[-1,3])
    lab_to_fxfyfz = tf.cast(tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ]),float)
    fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0],float), lab_to_fxfyfz)
    epsilon = 6.0/29.0
    linear_mask = tf.cast((fxfyfz_pixels <= epsilon),float)
    exponential_mask = tf.cast((fxfyfz_pixels > epsilon),float)
    xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29.0)) * linear_mask + ((fxfyfz_pixels+0.000001) ** 3) * exponential_mask
    xyz_pixels =tf.math.multiply(xyz_pixels, tf.constant([0.950456, 1.0, 1.088754],float))
    xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ],float)
    rgb_pixels =  tf.matmul(xyz_pixels, xyz_to_rgb)
    rgb_pixels = tf.clip_by_value(rgb_pixels,0, 1)
    inear_mask = tf.cast((rgb_pixels <= 0.0031308),float)
    exponential_mask = tf.cast((rgb_pixels > 0.0031308),float)
    srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (((rgb_pixels+0.000001) ** (1/2.4) * 1.055) - 0.055) * exponential_mask
    return tf.reshape(srgb_pixels, lab.shape)

