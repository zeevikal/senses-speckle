from tensorflow.keras.layers import AveragePooling2D, MaxPool2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv3D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model

from sklearn.utils import class_weight
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import matplotlib
import argparse
import random
import pickle
import glob
import cv2
import os

np.random.seed(42)
matplotlib.use("Agg")

# Data collection dates
DATES = ['20200114', '20200218']
# Data labels
SAMPLES = ['Zeev', 'Sergey', 'Yafim', 'Aviya']
SENSES = ['Smell', 'Hearing', 'Taste']
# Frames images size
FRAME_SIZE = 32

# Training params
BATCH_SIZE = 64
EPOCHS = 30

data_path = '../data/*'


def prep_train_val_frames():
    '''
    create training and validation sets
    :return:
    '''
    train_data = []
    train_labels = []
    val_data = []
    val_labels = []
    for train_val_dir in glob.glob(data_path):
        for sample in SAMPLES:
            for sense in SENSES:
                sense_frames_path = '{}/{}/{}/frames'.format(train_val_dir, sample, sense)
                no_sense_frames_path = '{}/{}/No_{}/frames'.format(train_val_dir, sample, sense)
                for frame_filename in tqdm(glob.glob('{}/*'.format(sense_frames_path))):
                    if os.path.isfile(frame_filename):
                        image = cv2.imread(frame_filename)
                        if 'train' in train_val_dir:
                            train_data.append(image)
                            train_labels.append('sense')
                        elif 'val' in train_val_dir:
                            val_data.append(image)
                            val_labels.append('sense')
                for frame_filename in tqdm(glob.glob('{}/*'.format(no_sense_frames_path))):
                    if os.path.isfile(frame_filename):
                        image = cv2.imread(frame_filename)
                        if 'train' in train_val_dir:
                            train_data.append(image)
                            train_labels.append('no_sense')
                        elif 'val' in train_val_dir:
                            val_data.append(image)
                            val_labels.append('no_sense')
    train_data = np.array(train_data)
    train_labels_orig = np.array(train_labels)
    val_data = np.array(val_data)
    val_labels_orig = np.array(val_labels)
    return train_data, train_labels_orig, val_data, val_labels_orig


def load_exist_frames_data(train_path='../data/raw_train_data_frames.npz', val_path='../data/raw_val_data_frames.npz',
                           train_data=None, train_labels_orig=None, val_data=None, val_labels_orig=None):
    '''
    load dataset if exists, else save a new dataset
    :param train_data:
    :param train_labels_orig:
    :param val_data:
    :param val_labels_orig:
    :return:
    '''
    if os.path.exists(train_path):
        train_file = np.load(train_path)
        train_data = train_file['x']
        train_labels_orig = train_file['y']
    else:
        np.savez_compressed(train_path, x=train_data, y=train_labels_orig)
    if os.path.exists(val_path):
        val_file = np.load(val_path)
        val_data = val_file['x']
        val_labels_orig = val_file['y']
    else:
        np.savez_compressed(val_path, x=val_data, y=val_labels_orig)
    return train_data, train_labels_orig, val_data, val_labels_orig


def create_64frames_set(x_orig, y_orig, inner_batch_cnt=64):
    x_data = list()
    y_data = list()
    for i in tqdm(range(0, len(x_orig) - 1)):
        inner_cnt = 0
        inner_batch = list()
        while inner_cnt < inner_batch_cnt:
            if y_orig[i] != y_orig[i + 1]: break
            im1 = x_orig[i]
            im2 = x_orig[i + 1]
            cls = y_orig[i]
            image = im2 - im1
            inner_batch.append(image)
            inner_cnt += 1
        if len(inner_batch) > 0:
            x_data.append(np.array(inner_batch))
            if 'no_sense' in cls:
                y_data.append(0)
            else:
                y_data.append(1)
        return x_data, y_data


def create_64frames_dataset(train_data, train_labels_orig, val_data, val_labels_orig):
    x_train, y_train = create_64frames_set(train_data, train_labels_orig)
    x_val, y_val = create_64frames_set(val_data, val_labels_orig)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    return x_train, y_train, x_val, y_val


def load_exist_64frames_data(train_path='../data/movement_train_data_frames.npz',
                             val_path='../data/movement_val_data_frames.npz',
                             x_train=None, y_train=None, x_val=None, y_val=None):
    '''
    load dataset if exists, else save a new dataset
    :param train_data:
    :param train_labels_orig:
    :param val_data:
    :param val_labels_orig:
    :return:
    '''
    if os.path.exists(train_path):
        train_file = np.load(train_path)
        x_train = train_file['x']
        y_train = train_file['y']
    else:
        np.savez_compressed(train_path, x=x_train, y=y_train)
    if os.path.exists(val_path):
        val_file = np.load(val_path)
        x_val = val_file['x']
        y_val = val_file['y']
    else:
        np.savez_compressed(val_path, x=x_val, y=y_val)
    return x_train, y_train, x_val, y_val


def labels_binarizer(y_train, y_val):
    lb = LabelBinarizer()
    train_labels = lb.fit_transform(y_train)
    val_labels = lb.fit_transform(y_val)
    return lb.classes_, train_labels, val_labels


def model(classes):
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), return_sequences=False, data_format="channels_last",
                         input_shape=(None, FRAME_SIZE, FRAME_SIZE, 3)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(len(classes), activation="softmax"))
    # opt = SGD(lr=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer="Adam", metrics=["accuracy"])
    return model


def train(x_train, y_train, x_val, y_val):
    checkpoint_filepath = '../models/sense_no_sense_classifier_checkpoint'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True,
                                                monitor='val_accuracy',
                                                mode='max', save_best_only=True)
    H = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=BATCH_SIZE,
                  epochs=EPOCHS, callbacks=[model_checkpoint_callback])
    return H
