import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
import os
import random
import time
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

ACTIONS = ["sense", "no_sense"]
reshape = (-1, 3000, 4) # 4 channels OpenBCI EEG device

# training params
epochs = 50
batch_size = 512

def create_data(starting_dir="../raw_data/20200913 EEG", is_train=True):
    training_data = {}
    for action in ACTIONS:
        if action not in training_data:
            training_data[action] = []

    for path, subdirs, files in os.walk(starting_dir):
        random.shuffle(files)
        if is_train:
            files = files[:6]
            print(files)
        else:
            files = files[6:]
            print(files)
        for name in files:
            data = np.load(os.path.join(path, name))
            if ACTIONS[1] in name or 'no_sesnse' in name or 'nosesnse' in name: # no_sense
                training_data[ACTIONS[1]].append(data)
            else:
                training_data[ACTIONS[0]].append(data)

    lengths = [len(training_data[action]) for action in ACTIONS]
    print(lengths)

    for action in ACTIONS:
        np.random.shuffle(training_data[action])  # note that regular shuffle is GOOF af
        training_data[action] = training_data[action][:min(lengths)]
    lengths = [len(training_data[action]) for action in ACTIONS]
    print(lengths)
    # creating X, y
    combined_data = []
    for action in ACTIONS:
        for data in training_data[action]:

            if action == "no_sense":
                combined_data.append([data, [1, 0]])
            else:
                combined_data.append([data, [0, 1]])

    np.random.shuffle(combined_data)
    print("length:",len(combined_data))
    return combined_data

def prep_data(data_path="../raw_data/EEG_train", is_train=True):
    print("creating training data")
    data = create_data(starting_dir=data_path, is_train=is_train)
    data_X = []
    data_y = []
    for X, y in data:
        data_X.append(X)
        data_y.append(y)
    return data_X, data_y

def prep_dataset():
    train_X, train_y = prep_data(data_path="../raw_data/EEG_train", is_train=True)
    test_X, test_y = prep_data(data_path="../raw_data/EEG_test", is_train=False)
    train_X = np.array(train_X).reshape(reshape)
    test_X = np.array(test_X).reshape(reshape)
    train_y = np.array(train_y)
    test_y = np.array(test_y)
    return train_X, train_y, test_X, test_y

def eeg_model():
    model = Sequential()

    model.add(Conv1D(512, (3), input_shape=train_X.shape[1:]))
    model.add(Activation('relu'))

    model.add(Conv1D(128, (2)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=(2)))

    model.add(Conv1D(64, (2)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=(2)))

    model.add(Flatten())

    model.add(Dense(512))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def train(model, train_X, train_y, test_X, test_y, model_path='../models/eeg_sense_nosense_model.h5'):
    h = model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs, validation_data=(test_X, test_y))
    model.save(model_path)
    return h

