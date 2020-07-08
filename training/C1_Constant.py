"""
Created on 22/05/2020

@author: Joshua Bensemann
"""
from silence_tensorflow import silence_tensorflow

silence_tensorflow()

import tensorflow as tf
from os import makedirs, path
from progress.bar import Bar
from models.neural_nets import AidedConvModel
from utilities.logging import append_list_as_row

BATCH_SIZE = 128
BATCHES_PER_EPOCH = 1000
BUFFER_SIZE = 100000
NUM_EPOCHS = 500
STARTING_THRESHOLD = .99

SAVE_PATH = 'logs/'
OUTPUT_FILE = SAVE_PATH + "Constant_Model1.csv"

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


def main(run, dataset_dict, print_messages=False):
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

    prefix = "Constant_Model1"

    current_dataset = tf.data.experimental.sample_from_datasets([image_train, sound_train]).batch(BATCH_SIZE).prefetch(
        2)
    current_dataset = iter(current_dataset)

    aided_model = AidedConvModel(num_outputs=60)
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
                aid = tf.ones(y.shape, dtype=x.dtype)

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
            aid = tf.ones(y.shape, dtype=x.dtype)
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
            aid = tf.ones(y.shape, dtype=x.dtype)
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


if __name__ == "__main__":
    print('Constant_Model1 training script')
