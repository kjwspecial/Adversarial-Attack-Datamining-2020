from rgb_lab_formulation_tf2 import *
import cv2
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
import copy

def processImage(x):
    x = tf.Variable(x)
    x = tf.expand_dims(x,0)
    return x

# L의 디테일 강화
def detail_enhance_lab(img, smooth_img):
    # 디테일 강화 parameter
    val0 = 15
    val2 = 1
    exposure = 1.0
    gamma = 1.0
    img = tf.squeeze(img,axis=0)
    smooth_img = tf.squeeze(smooth_img,axis=0)

    img_lab=rgb_to_lab(img)    
    smooth_img_lab=rgb_to_lab(smooth_img)
    
    img_l, img_a, img_b =img_lab[:,:,0], img_lab[:,:,1] ,img_lab[:,:,2]
    smooth_l, smooth_a, smooth_b = smooth_img_lab[:,:,0], smooth_img_lab[:,:,1] ,smooth_img_lab[:,:,2]
    
    #원래 이미지 - 스무딩된 이미지 (img_l-smooth_l) : 둘의 차이가 detail이라고 볼 수 있음.
    diff = sig((img_l-smooth_l)/100.0,val0)*100.0
    # 스무딩 된 이미지 조절
    base = (sig( (exposure*smooth_l-56.0)/100.0,val2 ) *100.0)+56.0
    # ress :디테일 강화 이미지 = 디테일 강화 + 스무딩(structure 보존)
    res = base + diff
    img_l = res
    # a,b는 그대로 반환
    img_a = img_a
    img_b = img_b
    img_lab = tf.stack([img_l, img_a, img_b], axis= 2)

    # Lab to RGB
    L_chan, a_chan, b_chan = preprocess_lab(img_lab)
    img_lab = deprocess_lab(L_chan, a_chan, b_chan)
    img_final = lab_to_rgb(img_lab)

    return img_final

def sig(x,a):
    y = 1./(1+tf.math.exp(-a*x)) - 0.5
    y05 = 1./(1+tf.math.exp(-tf.constant(a*0.5))) - 0.5
    y = y*(0.5/y05)
    return y


def recreate_image(image):
    recreated_im = image.numpy()
    recreated_im = tf.clip_by_value(recreated_im,0,1)
    recreated_im = np.round(recreated_im * 255)
    recreated_im = recreated[..., ::-1]
    return recreated_im


def PreidictLabel(x, model):
    logit, prob = model(x)
    class_x = tf.argmax(prob,axis=1)
    return class_x , logit
     
def AdvLoss(logits, target, num_classes = 10, kappa=0.0):
    target_one_hot = tf.eye(10)[target[0].numpy()]
    real = tf.reduce_sum(target_one_hot*logits,1)
    other = tf.math.reduce_max((1-target_one_hot)*logits - (target_one_hot*10000))
    return tf.math.maximum((real-other),kappa)
