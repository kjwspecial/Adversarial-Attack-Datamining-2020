import tensorflow as tf
from tensorflow.keras import models, layers
from filter.guided_filter import FastGuidedFilter
      
        
        
class AdaptiveNorm(models.Model):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = tf.Variable(tf.constant([1.0]))
        self.w_1 =  tf.Variable(tf.constant([0.0]))

        self.bn  = layers.BatchNormalization(momentum=0.999, epsilon=0.001)

    def call(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)
    
# FCNN  - 원래 목적은 주변 픽셀들 보간, layer가 너무 깊어도 학습이 어려움
def build_lr_net(norm=AdaptiveNorm, num_layers=5):
    layer = [
        layers.Conv2D(24,kernel_size=3 ,strides=(1,1), padding = 'SAME',dilation_rate=1),
        AdaptiveNorm(24),
        layers.LeakyReLU(0.2)
    ]
    for l in range(1, num_layers):
        layer += [
            layers.Conv2D(24,kernel_size=3 ,strides=(1,1), padding = 'SAME',dilation_rate=2**l),
            AdaptiveNorm(24),
            layers.LeakyReLU(0.2)
        ]
    layer += [
        layers.Conv2D(24,kernel_size=3 ,strides=(1,1), padding = 'SAME',dilation_rate=1),
        AdaptiveNorm(24),
        layers.LeakyReLU(0.2),
        layers.Conv2D(3,kernel_size=1 ,strides=(1,1), padding = 'SAME',dilation_rate=1),
    ]
    net = tf.keras.Sequential(layer)


    return net
class DeepGuidedFilter(models.Model):
    def __init__(self, radius=1, eps=1e-8):
        super(DeepGuidedFilter, self).__init__()
        self.lr = build_lr_net()
        self.gf = FastGuidedFilter(radius, eps)
    # x_lr : 입력 저해상도 이미지
    # x_hr : 입력 고해상도 이미지
    # self.lr(x_lr) : 출력 저해상도 이미지
    def call(self, x_lr, x_hr):
        return  tf.clip_by_value(self.gf(x_lr, self.lr(x_lr), x_hr),0,1)
