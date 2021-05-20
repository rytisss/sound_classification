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
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Data
# train
train_dir = r'C:\Users\Rytis\Desktop\sound_classification\archive\train/'
# test
test_dir = r'C:\Users\Rytis\Desktop\sound_classification\archive\test/'

# Directory for weight saving (creates if it does not exist)
weights_output_dir = r'output_label_smooth/'
weights_output_name = '5_down'
# batch size. How many samples you want to feed in one iteration?
batch_size = 32
# number_of_epoch. How many epochs you want to train?
number_of_epoch = 10

# input channels
input_channels = 4
# signal length
signal_length = 80999


def signal_classification_model(start_kernels=16, number_of_classes=11, input_shape=(signal_length, input_channels),
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

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
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
    step = epoch // 3
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

    def get_class_id_by_name(self, path):
        for i, class_name in enumerate(self.class_names):
            # get last folder from path
            last_folder = os.path.basename(os.path.dirname(path))
            if class_name.lower() in last_folder.lower():
                return i
        raise  # code should not reach this point... if it happens folder structure might be different than in description

    def get_labels(self):
        y = []
        for i in range(len(self.filenames)):
            y.append(self.get_class_id_by_name(self.filenames[i]))
        y = tf.one_hot(y, len(self.class_names))
        return y

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


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def make_confusion_matrix(model, test_generator):
    print('Predicting...')
    true_labels = test_generator.get_labels()
    y_true = np.argmax(true_labels, axis=1)
    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    conf_matrix = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(conf_matrix, test_generator.class_names)
    print(conf_matrix)
   """ fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_matrix)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1));
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_xticklabels(test_generator.class_names)

    ax.set_yticklabels(test_generator.class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    print(classification_report(y_true, y_pred, target_names=test_generator.class_names))
    print('names:')
    print(test_generator.class_names)"""


def train():
    # sample_rate, sound_file = scipy.io.wavfile.read(
    #    r'C:\src\Projects\sound_classification\data/kitchen_2_Dishes_1_4_white_80.wav')
    # normalized_signal = normalize(sound_file)

    # get all classes folder in dir
    classes = get_classes_names(train_dir)
    train_files = get_all_audio_files(train_dir)
    test_files = get_all_audio_files(test_dir)

    # Define model
    model = signal_classification_model(pretrained_weights=r'C:\src\Projects\sound_classification\output/_best.hdf5')

    train_generator = data_flow(train_files, batch_size, classes)
    test_generator = data_flow(test_files, batch_size, classes)

    # TODO: delete/comment following
    make_confusion_matrix(model, test_generator)

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
