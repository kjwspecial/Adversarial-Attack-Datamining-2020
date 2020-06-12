import tensorflow as tf
from tensorflow.keras import models, layers


class Model(models.Model):
    def __init__(self,base_model):
        super(Model,self).__init__()
        self.base_model = base_model
        self.top_layer = models.Sequential([
            layers.Dense(10)
            #layers.Activation(tf.nn.softmax)
        ])
        
    def call(self,inputs,training=False):
        x = self.base_model(inputs, training=training)
        x = layers.Flatten()(x)
        logits = self.top_layer(x, training=training)
        return logits, tf.nn.softmax(logits)