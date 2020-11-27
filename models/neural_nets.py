"""
Created on 21/05/2020

@author: Joshua Bensemann
"""
import tensorflow as tf
from tensorflow import keras

class ConvModel(keras.Model):

    def __init__(self, num_outputs):

        super(ConvModel, self).__init__()
        self.learning_rate = .00046155592162824545

        self.conv1 = keras.layers.Conv2D(32, 3, activation='relu')
        self.max1 = keras.layers.MaxPool2D()

        self.conv2 = keras.layers.Conv2D(64, 3, activation='tanh')
        self.max2 = keras.layers.MaxPool2D()

        self.conv3 = keras.layers.Conv2D(16, 3, activation='sigmoid')
        self.max3 = keras.layers.MaxPool2D()
        self.flat = keras.layers.Flatten()

        self.dense1 = keras.layers.Dense(512)
        self.dense2 = keras.layers.Dense(288)
        self.output_layer = keras.layers.Dense(num_outputs, activation='softmax')

    @tf.function
    def call(self, x, return_raw=False):
        x = self.conv1(x)
        x = self.max1(x)

        x = self.conv2(x)
        x = self.max2(x)

        x = self.conv3(x)
        x = self.max3(x)
        x = self.flat(x)

        raw_1 = self.dense1(x)
        x = keras.activations.relu(raw_1)

        raw_2 = self.dense2(x)
        x = keras.activations.tanh(raw_2)

        x = self.output_layer(x)

        if return_raw:
            return x, [raw_1, raw_2]

        else:
            return x


class NullConvModel(keras.Model):

    def __init__(self, num_outputs):

        super(NullConvModel, self).__init__()
        self.learning_rate = .00046155592162824545

        self.conv1 = keras.layers.Conv2D(32, 3, activation='relu')
        self.max1 = keras.layers.MaxPool2D()

        self.conv2 = keras.layers.Conv2D(64, 3, activation='tanh')
        self.max2 = keras.layers.MaxPool2D()

        self.conv3 = keras.layers.Conv2D(16, 3, activation='sigmoid')
        self.max3 = keras.layers.MaxPool2D()
        self.flat = keras.layers.Flatten()

        self.dense1 = keras.layers.Dense(513)
        self.dense2 = keras.layers.Dense(288)
        self.output_layer = keras.layers.Dense(num_outputs, activation='softmax')

    @tf.function
    def call(self, x, return_raw=False):
        x = self.conv1(x)
        x = self.max1(x)

        x = self.conv2(x)
        x = self.max2(x)

        x = self.conv3(x)
        x = self.max3(x)
        x = self.flat(x)

        raw_1 = self.dense1(x)
        x = keras.activations.relu(raw_1)

        raw_2 = self.dense2(x)
        x = keras.activations.tanh(raw_2)

        x = self.output_layer(x)

        if return_raw:
            return x, [raw_1, raw_2]

        else:
            return x


class AidedConvModel(keras.Model):

    def __init__(self, num_outputs):

        super(AidedConvModel, self).__init__()
        self.learning_rate = .00046155592162824545

        self.conv1 = keras.layers.Conv2D(32, 3, activation='relu')
        self.max1 = keras.layers.MaxPool2D()

        self.conv2 = keras.layers.Conv2D(64, 3, activation='tanh')
        self.max2 = keras.layers.MaxPool2D()

        self.conv3 = keras.layers.Conv2D(16, 3, activation='sigmoid')
        self.max3 = keras.layers.MaxPool2D()
        self.flat = keras.layers.Flatten()

        self.dense1 = keras.layers.Dense(512, activation='relu')
        self.dense2 = keras.layers.Dense(288, activation='tanh')
        self.output_layer = keras.layers.Dense(num_outputs, activation='softmax')


    @tf.function
    def call(self, x, aid_info):
        x = self.conv1(x)
        x = self.max1(x)

        x = self.conv2(x)
        x = self.max2(x)

        x = self.conv3(x)
        x = self.max3(x)
        x = self.flat(x)
        x = tf.concat([x, aid_info], 1)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.output_layer(x)

        return x

class NM_AidedConvModel(keras.Model):

    def __init__(self, num_outputs):

        super(NM_AidedConvModel, self).__init__()
        self.learning_rate = .00046155592162824545

        self.conv1 = keras.layers.Conv2D(32, 3, activation='relu')
        self.max1 = keras.layers.MaxPool2D()

        self.conv2 = keras.layers.Conv2D(64, 3, activation='tanh')
        self.max2 = keras.layers.MaxPool2D()

        self.conv3 = keras.layers.Conv2D(16, 3, activation='sigmoid')
        self.max3 = keras.layers.MaxPool2D()
        self.flat = keras.layers.Flatten()

        self.dense1 = keras.layers.Dense(512, activation='relu')
        self.dense2 = keras.layers.Dense(288, activation='tanh')
        self.neuro_mod = keras.layers.Dense(288, activation='sigmoid')
        self.output_layer = keras.layers.Dense(num_outputs, activation='softmax')


    @tf.function
    def call(self, x, aid_info):
        x = self.conv1(x)
        x = self.max1(x)

        x = self.conv2(x)
        x = self.max2(x)

        x = self.conv3(x)
        x = self.max3(x)
        x = self.flat(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = tf.multiply(x, self.neuro_mod(aid_info))

        x = self.output_layer(x)

        return x

###Concept Models
class EX2ConvModel(keras.Model):

    def __init__(self, num_outputs):

        super(EX2ConvModel, self).__init__()
        self.learning_rate = .0005864360116166073

        self.conv1 = keras.layers.Conv2D(16, 3, activation='tanh')
        self.max1 = keras.layers.MaxPool2D()

        self.conv2 = keras.layers.Conv2D(16, 3, activation='relu')
        self.max2 = keras.layers.MaxPool2D()

        self.conv3 = keras.layers.Conv2D(32, 3, activation='relu')
        self.max3 = keras.layers.MaxPool2D()
        self.flat = keras.layers.Flatten()

        self.dense1 = keras.layers.Dense(320)
        self.dense2 = keras.layers.Dense(448)
        self.output_layer = keras.layers.Dense(num_outputs, activation='softmax')

    @tf.function
    def call(self, x, return_raw=False):
        x = self.conv1(x)
        x = self.max1(x)

        x = self.conv2(x)
        x = self.max2(x)

        x = self.conv3(x)
        x = self.max3(x)
        x = self.flat(x)

        raw_1 = self.dense1(x)
        x = keras.activations.relu(raw_1)

        raw_2 = self.dense2(x)
        x = keras.activations.sigmoid(raw_2)

        x = self.output_layer(x)

        if return_raw:
            return x, [raw_1, raw_2]

        else:
            return x


class EX2AidedConvModel(keras.Model):

    def __init__(self, num_outputs):

        super(EX2AidedConvModel, self).__init__()
        self.learning_rate = .0005864360116166073

        self.conv1 = keras.layers.Conv2D(16, 3, activation='tanh')
        self.max1 = keras.layers.MaxPool2D()

        self.conv2 = keras.layers.Conv2D(16, 3, activation='relu')
        self.max2 = keras.layers.MaxPool2D()

        self.conv3 = keras.layers.Conv2D(32, 3, activation='relu')
        self.max3 = keras.layers.MaxPool2D()
        self.flat = keras.layers.Flatten()

        self.dense1 = keras.layers.Dense(320, activation='relu')
        self.dense2 = keras.layers.Dense(448, activation='sigmoid')
        self.output_layer = keras.layers.Dense(num_outputs, activation='softmax')


    @tf.function
    def call(self, x, aid_info):
        x = self.conv1(x)
        x = self.max1(x)

        x = self.conv2(x)
        x = self.max2(x)

        x = self.conv3(x)
        x = self.max3(x)
        x = self.flat(x)
        x = tf.concat([x, aid_info], 1)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.output_layer(x)

        return x