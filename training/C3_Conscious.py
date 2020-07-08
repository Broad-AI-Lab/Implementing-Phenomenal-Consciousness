"""
Created on 22/05/2020

@author: Joshua Bensemann
"""
from silence_tensorflow import silence_tensorflow

silence_tensorflow()

import tensorflow as tf
import numpy as np
from os import makedirs, path
from progress.bar import Bar
from models.neural_nets import EX2ConvModel, EX2AidedConvModel
from utilities.logging import append_list_as_row
from models.simple_learners import HebbianLearner

BATCH_SIZE = 128
BATCHES_PER_EPOCH = 1000
BUFFER_SIZE = 100000
NUM_EPOCHS = 500
STARTING_THRESHOLD = .99

SAVE_PATH = 'logs/'
OUTPUT_FILE = SAVE_PATH + "Conscious_Model3.csv"

if not path.isdir(SAVE_PATH):
    makedirs(SAVE_PATH)

if not path.exists(OUTPUT_FILE):
    append_list_as_row(OUTPUT_FILE, ["Run", "Epoch", "Loss", "Accuracy",
                                     "Val_Loss", "Val Accuracy",
                                     "Image_Loss", "Image Accuracy",
                                     "Sound_Loss", "Sound Accuracy",
                                     "Hebb Accuracy", "Hebb Number", 'Pretraining Epochs'])

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()


def loss(model, x, y, aid):
    if aid is None:
        y_, outputs = model.call(x, return_raw=True)
    else:
        y_ = model.call(x, aid)
        outputs = None

    return loss_object(y_true=y, y_pred=y_), outputs


def grad(model, inputs, targets, aid):
    with tf.GradientTape() as tape:
        loss_value, outputs = loss(model, inputs, targets, aid)

    return loss_value, tape.gradient(loss_value, model.trainable_variables), outputs


def train_hebb(hebb1, hebb2, outputs, sup_y):
    hebb1.train(outputs[0], sup_y)
    hebb2.train(outputs[1], sup_y)


def main(run, dataset_dict, print_messages=False):
    @tf.function
    def train_step(model, x, y, optimizer):
        loss_value, grads, outputs = grad(model, x, y, None)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss_value, outputs

    @tf.function
    def aid_train_step(model, x, y, optimizer, aid=None):
        loss_value, grads, _ = grad(model, x, y, aid)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss_value

    image_train = dataset_dict['image_train'].shuffle(BUFFER_SIZE).repeat()
    sound_train = dataset_dict['sound_train'].shuffle(BUFFER_SIZE).repeat()
    image_test = dataset_dict['image_test']
    sound_test = dataset_dict['sound_test']

    image_test = image_test.batch(BATCH_SIZE)
    sound_test = sound_test.batch(BATCH_SIZE)

    prefix = "Conscious_Model3"
    model = EX2ConvModel(num_outputs=60)
    optimizer = tf.keras.optimizers.Adam(learning_rate=model.learning_rate)

    hebb1 = HebbianLearner(model.layers[-3].units)
    hebb2 = HebbianLearner(model.layers[-2].units)

    current_dataset = tf.data.experimental.sample_from_datasets([image_train, sound_train]).batch(BATCH_SIZE).prefetch(
        2)
    current_dataset = iter(current_dataset)

    hebb1_avg = tf.keras.metrics.Mean()
    hebb2_avg = tf.keras.metrics.Mean()

    hebb_threshold = STARTING_THRESHOLD
    hebb_aid = None
    hebb_number = 0
    aid_accuracy = 0.
    pretrain_epochs = 0

    #Training aid for aided model
    for epoch in range(NUM_EPOCHS):
        hebb1_avg.reset_states()
        hebb2_avg.reset_states()

        hebb1.reset()
        hebb2.reset()

        with Bar('{} Run {:03d} Epoch {:04}'.format('Pre-training', run, epoch + 1), max=BATCHES_PER_EPOCH) as bar:
            for i in range(BATCHES_PER_EPOCH):
                x, y = next(current_dataset)
                loss_value, raw = train_step(model, x, y[:,0], optimizer)
                train_hebb(hebb1, hebb2, raw, y[:,1])
                bar.next()

        # Evaluation training loss seperately between tasks
        for x, y in image_test:
            loss_value, raw = loss(model, x, y[:,0], None)

            hebb1_avg(hebb1.score(raw[0], y[:,1]))
            hebb2_avg(hebb2.score(raw[1], y[:,1]))

        for x, y in sound_test:
            loss_value, raw = loss(model, x, y[:,0], None)

            hebb1_avg(hebb1.score(raw[0], y[:,1]))
            hebb2_avg(hebb2.score(raw[1], y[:,1]))

        hebb1_acc = float(hebb1_avg.result())
        hebb2_acc = float(hebb2_avg.result())

        pretrain_epochs = epoch + 1
        if hebb1_acc > aid_accuracy:
            hebb_aid = hebb1
            hebb_number = 1
            aid_accuracy = hebb1_acc

        if hebb2_acc > aid_accuracy:
            hebb_aid = hebb2
            hebb_number = 2
            aid_accuracy = hebb2_acc

        if print_messages:
            print("Hebb1 Accuracy {:.3%}, Hebb2 Accuracy {:.3%}, Threshold {:.3%}".format(hebb1_acc, hebb2_acc, hebb_threshold))

        if aid_accuracy > hebb_threshold:
            break


    print("\rPretraining complete after {} epochs, with {:.3%} accuracy".format(pretrain_epochs, aid_accuracy))

    aided_model = EX2AidedConvModel(num_outputs=60)
    optimizer = tf.keras.optimizers.Adam(learning_rate=aided_model.learning_rate)

    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    val_loss_avg = tf.keras.metrics.Mean()
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    image_loss_avg = tf.keras.metrics.Mean()
    image_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    sound_loss_avg = tf.keras.metrics.Mean()
    sound_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    #main training
    for epoch in range(NUM_EPOCHS):
        epoch_loss_avg.reset_states()
        epoch_accuracy.reset_states()
        val_loss_avg.reset_states()
        val_accuracy.reset_states()
        image_loss_avg.reset_states()
        image_accuracy.reset_states()
        sound_loss_avg.reset_states()
        sound_accuracy.reset_states()

        with Bar('{} Run {:03d} Epoch {:04}'.format(prefix, run, epoch + 1), max=BATCHES_PER_EPOCH) as bar:
            for i in range(BATCHES_PER_EPOCH):
                x, y = next(current_dataset)
                _, raw = model.call(x, return_raw=True)
                aid = tf.reshape(hebb_aid.call(raw[hebb_number-1]), [-1,1])
                loss_value = aid_train_step(aided_model, x, y[:,0], optimizer, aid)
                epoch_loss_avg(loss_value)
                epoch_accuracy(y[:,0], aided_model(x, aid))
                bar.next()

        if print_messages:
            training_string = "\r{} Run {:03d} Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}...".format(
                prefix,
                run,
                epoch + 1,
                epoch_loss_avg.result(),
                epoch_accuracy.result()
            )

            print(training_string, end="")

        # Evaluation training loss seperately between tasks
        for x, y in image_test:
            _, raw = model.call(x, return_raw=True)
            aid = tf.reshape(hebb_aid.call(raw[hebb_number-1]), [-1,1])
            loss_value, _ = loss(aided_model, x, y[:,0], aid)

            image_loss_avg(loss_value)
            val_loss_avg(loss_value)

            predictions = aided_model(x, aid)
            image_accuracy(y[:,0], predictions)
            val_accuracy(y[:,0], predictions)

        if print_messages:
            image_string = "Image Validation Loss: {:.3f}, Accuracy: {:.3%}".format(image_loss_avg.result(),
                                                                                    image_accuracy.result())

            print(image_string, end="...")

        for x, y in sound_test:
            _, raw = model.call(x, return_raw=True)
            aid = tf.reshape(hebb_aid.call(raw[hebb_number-1]), [-1,1])
            loss_value, raw = loss(aided_model, x, y[:,0], aid)

            sound_loss_avg(loss_value)
            val_loss_avg(loss_value)

            predictions = aided_model(x, aid)
            sound_accuracy(y[:,0], predictions)
            val_accuracy(y[:,0], predictions)

        if print_messages:
            sound_string = "Sound Validation Loss: {:.3f}, Accuracy: {:.3%}".format(sound_loss_avg.result(),
                                                                                    sound_accuracy.result())

            print(sound_string, end="...\n")

        if print_messages:
            val_string = "Validation Loss: {:.3f}, Accuracy: {:.3%}".format(val_loss_avg.result(),
                                                                            val_accuracy.result())

            print(val_string, end="...\n")

        append_list_as_row(OUTPUT_FILE, [run, epoch + 1,
                                         float(epoch_loss_avg.result()), float(epoch_accuracy.result()),
                                         float(val_loss_avg.result()), float(val_accuracy.result()),
                                         float(image_loss_avg.result()), float(image_accuracy.result()),
                                         float(sound_loss_avg.result()), float(sound_accuracy.result()),
                                         float(aid_accuracy), int(hebb_number), int(pretrain_epochs)
                                         ])

    output_folder = './logs/output/'

    if not path.isdir(output_folder):
        makedirs(output_folder)

    test_output_file = './logs/output/{}_run{}_acc{:.3%}.csv'.format(prefix,
                                                                     run,
                                                                     val_accuracy.result()
                                                                     )

    for x, y in image_test:
        actual = y.numpy()[:,0]
        predictions = model(x)
        predictions = np.argmax(predictions.numpy(), axis=1)

        append_list_as_row(test_output_file, ['actual'] + list(actual))
        append_list_as_row(test_output_file, ['prediction'] + list(predictions))

    for x, y in sound_test:
        actual = y.numpy()[:,0]
        predictions = model(x)
        predictions = np.argmax(predictions.numpy(), axis=1)

        append_list_as_row(test_output_file, ['actual'] + list(actual))
        append_list_as_row(test_output_file, ['prediction'] + list(predictions))


if __name__ == "__main__":
    print('Conscious_Model3 training script')
