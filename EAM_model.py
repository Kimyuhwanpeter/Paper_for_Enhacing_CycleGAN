# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math
#import keras_contrib as tfa

from absl import flags

Conv2D = tf.keras.layers.Conv2D
Flatten = tf.keras.layers.Flatten
BatchNorm = tf.keras.layers.BatchNormalization
Con2D_trans = tf.keras.layers.Conv2DTranspose

L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

# 현재 Batch normalization을 instance normalization을 대체로 사용하였다. 내 경험상 GAN에서는 Batch normalization 보다 instance normalization이 효과적이라고 생각한다.
# 하지만!! 현재(2019/7/25)는 tensorflow 2.0 에서 instance normalization을 지원하지 않고 tensorflow_addons도 window를 지원하지 않아서 instance normalization을 사용할 수 없다.
################################################################################################################################

# ==============================================================================
# =                                  networks                                  =
# ==============================================================================


class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

class Pad(tf.keras.layers.Layer):

    def __init__(self, paddings, mode='CONSTANT', constant_values=0, **kwargs):
        super(Pad, self).__init__(**kwargs)
        self.paddings = paddings
        self.mode = mode
        self.constant_values = constant_values

    def call(self, inputs):
        return tf.pad(inputs, self.paddings, mode=self.mode, constant_values=self.constant_values)

def instance_norm(input):           # 불안정함(tensorflow 2.0 에서 제대로 instance normalization을 지원해줄 때 사용해야 할 것같다.)
    depth = input.get_shape()[3]
    #scale = tf.compat.v1.get_variable("scale", [depth], initializer=tf.compat.v1.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
    #offset = tf.compat.v1.get_variable("offset", [depth], initializer=tf.compat.v1.constant_initializer(0.0))
    scale = ()

    mean, variance = tf.nn.moments(input, axes=[1,2], keepdims=True)
    epsilon = 1e-5
    inv = tf.math.rsqrt(variance + epsilon)
    normalized = (input - mean) * inv
    
    return (scale * normalized + offset)       # 이 부분이 불안정한 것같다.

def ResnetGenerator_v2(input_shape=(256, 256, 3),
                    output_channels=3,
                    dim=64,
                    n_downsamplings=2,
                    n_blocks=9,
                    norm='instance_norm'):
    Norm = InstanceNormalization(epsilon=1e-5)
    
    def _residual_block(x):
        dim = x.shape[-1]
        h = x

        h = Pad([[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')(h)
        h = tf.keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = InstanceNormalization(epsilon=1e-5)(h)
        h = tf.keras.layers.ReLU()(h)

        h = Pad([[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')(h)
        h = tf.keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = InstanceNormalization(epsilon=1e-5)(h)

        return tf.keras.layers.add([x, h])

    def concat(image, label):
        if 3 == image.shape[3]:
            i_ = image
            i = label
            concat = tf.concat([i_, i], 3)
            image = concat
        elif 57 == image.shape[3]:
            image = image
            i_ = image[:,:,:,0:3]
        return image, i_
    
    h = inputs = tf.keras.Input(shape=input_shape)
    #l = inputs_label = tf.keras.Input(shape=real_label)

    #image, i_ = concat(h, l)
    
    # 0
    #h = inputs = tf.keras.Input(shape=input_shape)

    # 1
    h = Pad([[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')(h)
    layer1 = tf.keras.layers.Conv2D(dim, 7, padding='valid', use_bias=False)
    Conv1 = layer1(h)
    l1 = tf.convert_to_tensor(layer1.get_weights())
    h = InstanceNormalization(epsilon=1e-5)(Conv1)
    res1 = h
    h = tf.keras.layers.ReLU()(h)
    classmap1 = get_class_map(l1, h, 256)

    # 2
    #for _ in range(n_downsamplings):
    #    dim *= 2
    #    h = tf.keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
    #    h = InstanceNormalization(epsilon=1e-5)(h)
    #    h = tf.keras.layers.ReLU()(h)
    dim *= 2
    layer2 = tf.keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)
    Conv2 = layer2(h)
    l2 = tf.convert_to_tensor(layer2.get_weights())
    h = InstanceNormalization(epsilon=1e-5)(Conv2)
    res2 = h
    h = tf.keras.layers.ReLU()(h)
    classmap2 = get_class_map(l2, h, 256)

    dim *= 2
    layer3 = tf.keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)
    Conv3 = layer3(h)
    l3 = tf.convert_to_tensor(layer3.get_weights())
    h = InstanceNormalization(epsilon=1e-5)(Conv3)
    h = tf.keras.layers.ReLU()(h)
    classmap3 = get_class_map(l3, h, 256)

    # 3
    for _ in range(n_blocks):
        h = _residual_block(h)
    Conv4 = h

    # 4
    #for _ in range(n_downsamplings):
    #    dim //= 2
    #    h = tf.keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
    #    h = InstanceNormalization(epsilon=1e-5)(h)
    #    h = tf.keras.layers.ReLU()(h)
    dim //= 2
    layer4 = tf.keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)
    Conv5 = layer4(Conv4)
    l4 = tf.convert_to_tensor(layer4.get_weights())
    h = InstanceNormalization(epsilon=1e-5)(Conv5)
    res2_pair = h + res2
    h = tf.keras.layers.ReLU()(res2_pair)
    classmap4 = get_class_map(l4, h, 256)

    dim //= 2
    layer5 = tf.keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)
    Conv6 = layer5(h)
    l5 = tf.convert_to_tensor(layer5.get_weights())
    h = InstanceNormalization(epsilon=1e-5)(Conv6)
    res1_pair = h + res1
    h = tf.keras.layers.ReLU()(res1_pair)
    classmap5 = get_class_map(l5, h, 256)


    # 5
    h = Pad([[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')(h)
    layer6 = tf.keras.layers.Conv2D(output_channels, 7, padding='valid', name='visualized_layer')
    Conv7 = layer6(h)
    #l6 = tf.convert_to_tensor(layer6.get_weights())
    h = tf.keras.layers.add([Conv7, inputs])            # resdual 
    h = tf.keras.layers.Activation('tanh')(h)
    #classmap6 = get_class_map(l6, Conv7, 256)

    return tf.keras.Model(inputs=inputs, outputs=h)

def ConvDiscriminator_v2(input_shape=(256, 256, None),
                      dim=64,
                      n_downsamplings=3,
                      norm='instance_norm'):
    
    dim_ = dim
    Norm = InstanceNormalization(epsilon=1e-5)

    # 0
    h = inputs = tf.keras.Input(shape=input_shape)

    # 1
    Conv1 = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(Conv1)

    #for _ in range(n_downsamplings - 1):
    #    dim = min(dim * 2, dim_ * 8)
    #    h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
    #    h = InstanceNormalization(epsilon=1e-5)(h)
    #    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    dim = min(dim * 2, dim_ * 8)
    Conv2 = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
    h = InstanceNormalization(epsilon=1e-5)(Conv2)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    dim = min(dim * 2, dim_ * 8)
    Conv3 = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
    h = InstanceNormalization(epsilon=1e-5)(Conv3)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)


    # 2
    dim = min(dim * 2, dim_ * 8)
    Conv4 = tf.keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
    h = InstanceNormalization(epsilon=1e-5)(Conv4)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 3
    Conv5 = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)


    return tf.keras.Model(inputs=inputs, outputs=Conv5)


# ==============================================================================
# =                          learning rate scheduler                           =
# ==============================================================================

class LinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate

######################################################
# activation map function

def get_class_map(weight, conv, im_size):

    output_channels = int(conv.get_shape()[-1])
    conv_resized = tf.image.resize(conv, [im_size, im_size])
    w = tf.reshape(weight, [-1, 1, output_channels])

    conv_resized = tf.reshape(conv_resized, [-1, im_size * im_size, output_channels])
    classmap = tf.linalg.matmul(conv_resized, w, adjoint_b=True)
    classmap = tf.reshape(classmap, [-1, im_size, im_size])
    return classmap