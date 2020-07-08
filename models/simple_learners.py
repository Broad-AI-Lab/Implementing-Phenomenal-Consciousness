"""
Created on Tue Mar 31 11:55:53 2020

@author: Joshua Bensemann
"""
import tensorflow as tf
import pickle


def convert_data(data):
    data = tf.math.divide_no_nan(data, tf.math.abs(data))  # Divide data by its absolute value
    return tf.cast(data, dtype="float32")


class HebbianLearner:

    def __init__(self, num_inputs, learning_rate=.1):
        self.num_inputs = num_inputs
        self.learning_rate = tf.constant(learning_rate, dtype='float32')
        self.weights = tf.zeros(num_inputs)
        self.bias = tf.zeros(1)

    def reset(self):
        self.weights = tf.zeros(self.weights.shape[0])
        self.bias = tf.zeros(1)

    def save_weights(self, filepath):
        to_save = (self.weights, self.bias)
        pickle.dump(to_save, open(filepath, "wb"))
        print("Weights saved to {}".format(filepath))

    def load_weights(self, filepath):
        loaded = pickle.load(open(filepath, "rb"))
        self.weights, self.bias = loaded
        print("Weights loaded successfully")

    def train(self, x, y):
        y = convert_data(y)
        x = convert_data(x)

        weights_update = tf.einsum('ij,i->ij', x, y)  # entire row of x[i:;] multiplied by y[i]
        weights_update = tf.math.scalar_mul(self.learning_rate, weights_update)
        weights_update = tf.reduce_sum(weights_update, axis=0)
        self.weights = tf.math.add(self.weights, weights_update)

        bias_update = tf.math.reduce_sum(y)
        self.bias = tf.math.add(self.bias, bias_update)

    @tf.function
    def call(self, x, one_hot=False):
        x = convert_data(x)
        y = tf.math.add(tf.linalg.matvec(x, self.weights), self.bias)
        if one_hot:
            y_1 = tf.maximum(convert_data(y), 0.)
            y_2 = tf.math.subtract(1., y_1)
            y_1 = tf.reshape(y_1, (-1,1))
            y_2 = tf.reshape(y_2, (-1, 1))
            return tf.concat([y_1, y_2], axis=1)

        return convert_data(y)

    def score(self, x, y):
        x = convert_data(x)
        y = convert_data(y)

        y_pred = tf.math.add(tf.linalg.matvec(x, self.weights), self.bias)
        y_pred = convert_data(y_pred)

        acc = tf.keras.metrics.Accuracy()
        acc.update_state(y, y_pred)

        return float(acc.result())