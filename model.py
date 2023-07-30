import tensorflow as tf
from tensorflow.keras import layers as ksl
from matplotlib import pyplot as plt
import numpy as np
#from keras.optimizers import learning_rate_schedule
from keras.layers.attention.multi_head_attention import activation
class Model:
  def __init__(self):
    self.inputShape=[784]
    self.net=None
  def buildModel(self):
   self.net=tf.keras.Sequential([ksl.Dense(256,activation='tanh',input_shape=self.inputShape),
                              ksl.Dense(128,activation='tanh'),
                              ksl.Dense(64,activation='tanh'),
                              ksl.Dense(10,activation='softmax')
                              ])
  def compileModel(self):
    tf.keras.utils.plot_model(self.net,'model.png')
    self.net.summary()
    loss=tf.keras.losses.CategoricalCrossentropy()
    optim=tf.keras.optimizers.SGD(learning_rate=0.01)
    self.net.compile(loss=loss)
    #self.net.compile(loss='CategoricalCrossentropy')




model=Model()
model.buildModel()
model.compileModel()