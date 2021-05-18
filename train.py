import glob
import os
import tensorflow as tf
import random

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LeakyReLU, BatchNormalization, GlobalAveragePooling1D, Dense
from tensorflow.keras.optimizers import Adam
import scipy.io.wavfile

# Data
# train
train_dir = r'C:\src\Projects\sound_classification\data/'
# test
test_dir = r'C:\src\Projects\sound_classification\data/'

# Directory for weight saving (creates if it does not exist)
weights_output_dir = r'output/'
weights_output_name = '5_down'
# batch size. How many samples you want to feed in one iteration?
batch_size = 8
# number_of_epoch. How many epochs you want to train?
number_of_epoch = 40

# input channels
input_channels = 4
# signal length
signal_length = 80999


def signal_classification_model(start_kernels=16, number_of_classes=3, input_shape=(signal_length, input_channels),
                                pretrained_weights=None):
    # https://keras.io/api/layers/convolution_layers/convolution1d/
    model = tf.keras.Sequential([
        ############ 1
        Conv1D(start_kernels, kernel_size=7, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False,
               input_shape=input_shape),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling1D(pool_size=2, strides=2),  # <---1
        ############ 2
        Conv1D(start_kernels * 2, kernel_size=5, strides=2, padding='same', kernel_initializer='he_normal',
               use_bias=False,
               input_shape=input_shape),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling1D(pool_size=2, strides=2),  # <---2
        ############ 3
        Conv1D(start_kernels * 4, kernel_size=5, strides=2, padding='same', kernel_initializer='he_normal',
               use_bias=False,
               input_shape=input_shape),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling1D(pool_size=2, strides=2),  # <---3
        ############# 4
        Conv1D(start_kernels * 8, kernel_size=5, strides=2, padding='same', kernel_initializer='he_normal',
               use_bias=False,
               input_shape=input_shape),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling1D(pool_size=2, strides=2),  # <---4
        ############# 5
        Conv1D(start_kernels * 8, kernel_size=5, strides=2, padding='same', kernel_initializer='he_normal',
               use_bias=False,
               input_shape=input_shape),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling1D(pool_size=2, strides=2),  # <---5
        ############# fully connected
        GlobalAveragePooling1D(),
        Dense(128),
        LeakyReLU(alpha=0.1),
        Dense(32),
        LeakyReLU(alpha=0.1),
        Dense(number_of_classes, activation='softmax')
    ])

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=Adam(lr=1e-3),
                  metrics=['categorical_accuracy'])
    model.summary()
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model


class CustomSaver(tf.keras.callbacks.Callback):
    def __init__(self):
        self.best_val_score = 0.0

    def on_epoch_end(self, epoch, logs=None):
        # also save if validation error is smallest
        if 'val_categorical_accuracy' in logs.keys():
            val_score = logs['val_categorical_accuracy']
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                print('New best weights found!')
                self.model.save(weights_output_dir + '_best.hdf5')
        else:
            print('Key val_accuracy does not exist!')


# This function keeps the learning rate at 0.001 for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch):
    step = epoch // 12
    init_lr = 0.001
    lr = init_lr / 2 ** step
    print('Epoch: ' + str(epoch) + ', learning rate = ' + str(lr))
    return lr


def normalize(signal, min_val=-32768, max_val=32767):
    signal_float = signal.astype(np.float32)
    return np.interp(signal_float, (min_val, max_val), (-1.0, 1.0))


def get_last_folder_names(paths):
    folders = []
    for path in paths:
        # get folder inside, it will be class name
        folders_inside = glob.glob(path + '*/')
        folders.append(os.path.basename(os.path.normpath(folders_inside[0])))
    return sorted(folders)


def get_all_classes_folders(path):
    return glob.glob(path + '*/')


def get_classes_names(path):
    train_folder = get_all_classes_folders(path)
    classes = get_last_folder_names(train_folder)
    return classes


def get_all_audio_files(path):
    train_folders = get_all_classes_folders(path)
    files = []
    for train_folder in train_folders:
        files.extend(glob.glob(train_folder + '/**/*.wav', recursive=True))
    # shuffle
    random.shuffle(files)
    return files


class data_flow(Sequence):
    def __init__(self, filenames, batch_size, class_names):
        self.filenames = filenames
        self.batch_size = batch_size
        self.class_names = class_names

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def remap_to_8bit(self, image, min_val, max_val):
        remap_image = np.interp(image, (min_val, max_val), (0, 255))
        return remap_image.astype(np.uint8)

    def get_class_id_by_name(self, path):
        for i, class_name in enumerate(self.class_names):
            if class_name in path:
                return i
        raise

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        x = []
        y = []
        for filename in batch_x:
            sample_rate, sound_file = scipy.io.wavfile.read(filename)
            normalized_signal = normalize(sound_file)
            x.append(normalized_signal)
            y.append(self.get_class_id_by_name(filename))

        x = np.array(x)
        y = tf.one_hot(y, len(self.class_names))

        return x, y


def train():
    # sample_rate, sound_file = scipy.io.wavfile.read(
    #    r'C:\src\Projects\sound_classification\data/kitchen_2_Dishes_1_4_white_80.wav')
    # normalized_signal = normalize(sound_file)

    # get all classes folder in dir
    classes = get_classes_names(train_dir)
    train_files = get_all_audio_files(train_dir)
    test_files = get_all_audio_files(test_dir)

    # Define model
    model = signal_classification_model()

    train_generator = data_flow(train_files, batch_size, classes)
    test_generator = data_flow(test_files, batch_size, classes)

    # create weights output directory
    if not os.path.exists(weights_output_dir):
        print('Output directory doesnt exist!\n')
        print('It will be created!\n')
        os.makedirs(weights_output_dir)

    # Define template of each epoch weight name. They will be save in separate files
    weights_name = weights_output_dir + weights_output_name + "-{epoch:03d}-{loss:.4f}.hdf5"
    # Custom saving for the best-performing weights
    saver = CustomSaver()
    # Learning rate scheduler
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    # Make checkpoint for saving each
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(weights_name, monitor='loss', verbose=1, save_best_only=False,
                                                          save_weights_only=False)
    model.fit(train_generator,
              epochs=number_of_epoch,
              validation_data=test_generator,
              callbacks=[model_checkpoint, learning_rate_scheduler, saver])


if __name__ == "__main__":
    train()