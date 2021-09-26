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

SMALL_SIZE = 14
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Data collection dates
DATES = ['20200114', '20200218']
# Data labels
SAMPLES = ['Zeev', 'Sergey', 'Yafim', 'Aviya']
SENSES = ['Smell', 'Hearing', 'Taste']
# Frames images size
FRAME_SIZE = 32

data_path = '../data/*'


def plot_training_process(history='../models/sense_no_sense_ConvLSTM2D_classifier_30_epochs_20201002_HISTORY',
                          fig_path='../models/sense_no_sense_ConvLSTM2D_classifier_30_epochs_20201002_HISTORY.png',
                          epochs=30, to_show=False):
    with open(history, 'rb') as f:
        H = pickle.load(f)
        # plot the training loss and accuracy (!!! if read original H must add `H.history` !!!)
    N = epochs
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(fig_path)
    if to_show:
        plt.show()


def plot_model(model, fig_path="../models/model.png"):
    plot_model(model, to_file=fig_path, show_shapes=True, show_layer_names=True)


def validate_model(test_data, model_path='../models/sense_no_sense_ConvLSTM2D_classifier_30_epochs_20201002.h5',
                   batch_size=64, workers=-1, verbose=1):
    model = load_model(model_path)
    predictions = model.predict(test_data, batch_size=64, workers=-1, verbose=1)
    return predictions


def get_classification_report(y_val, predictions):
    print(classification_report(y_val, predictions.argmax(axis=1),
                                target_names=['no_sense', 'sense']))  # target_names=lb.classes_


def prep_for_validation(model_path='../models/sense_no_sense_ConvLSTM2D_classifier_30_epochs_20201002.h5',
                        inner_batch_cnt=64, subject_data_path='../data/subject_data.pickle',
                        subject_predictions_path='../data/subject_predictions.pickle'):
    model = load_model(model_path)
    subject_data = dict()
    subject_predictions = dict()
    for train_val_dir in glob.glob(data_path):
        if train_val_dir.split(sep=os.sep)[-1] == 'val':
            for sample in SAMPLES:
                subject_data[sample] = dict()
                for sense in SENSES:
                    subject_data[sample][sense] = dict()
                    subject_data[sample][sense]['x'] = list()
                    subject_data[sample][sense]['y'] = list()
                    sense_frames_path = '{}/{}/{}/frames'.format(train_val_dir, sample, sense)
                    no_sense_frames_path = '{}/{}/No_{}/frames'.format(train_val_dir, sample, sense)
                    for frame_filename in tqdm(glob.glob('{}/*'.format(sense_frames_path))):
                        if os.path.isfile(frame_filename):
                            image = cv2.imread(frame_filename)
                            subject_data[sample][sense]['x'].append(image)
                            subject_data[sample][sense]['y'].append(1)
                    for frame_filename in tqdm(glob.glob('{}/*'.format(no_sense_frames_path))):
                        if os.path.isfile(frame_filename):
                            image = cv2.imread(frame_filename)
                            subject_data[sample][sense]['x'].append(image)
                            subject_data[sample][sense]['y'].append(0)
    for subject, sdata in subject_data.items():
        for sense, ssdata in sdata.items():
            subject_data[subject][sense]['subtracted_x'] = list()
            subject_data[subject][sense]['subtracted_y'] = list()
            for i in tqdm(range(0, len(ssdata['x']) - 1)):
                inner_cnt = 0
                inner_batch = list()
                while inner_cnt < inner_batch_cnt:
                    if ssdata['y'][i] != ssdata['y'][i + 1]: break
                    im1 = ssdata['x'][i]
                    im2 = ssdata['x'][i + 1]
                    cls = ssdata['y'][i]
                    image = im2 - im1
                    inner_batch.append(image)
                    inner_cnt += 1
                if len(inner_batch) > 0:
                    subject_data[subject][sense]['subtracted_x'].append(np.array(inner_batch))
                    subject_data[subject][sense]['subtracted_y'].append(cls)
    for subject, sdata in subject_data.items():
        for sense, ssdata in sdata.items():
            model_pred = model.predict(np.array(ssdata['subtracted_x']), batch_size=64, workers=-1, verbose=1)
            subject_data[subject][sense]['model_pred'] = model_pred
    for subject, sdata in subject_data.items():
        subject_predictions[subject] = dict()
        for sense, ssdata in sdata.items():
            subject_predictions[subject][sense] = dict()
            sense_preds = list()
            no_sense_preds = list()
            for i, t in enumerate(ssdata['subtracted_y']):
                if t == 0:
                    no_sense_preds.append(ssdata['model_pred'][i].argmax())
                else:
                    sense_preds.append(ssdata['model_pred'][i].argmax())
            subject_predictions[subject][sense]['sense_mean'] = np.mean(sense_preds)
            subject_predictions[subject][sense]['no_sense_mean'] = np.mean(no_sense_preds)
            subject_predictions[subject][sense]['sense_std'] = np.var(sense_preds)
            subject_predictions[subject][sense]['no_sense_std'] = np.var(no_sense_preds)
    with open(subject_data_path, 'wb') as f:
        pickle.dump(subject_data, f)
    with open(subject_predictions_path, 'wb') as f:
        pickle.dump(subject_predictions, f)
    return subject_data, subject_predictions


def ml_pred_bar_plot_per_subject(subject_predictions, fig_path='../models/ml_pred_bar_plot_per_subject.png',
                                 to_show=False):
    N = 5

    smell_means = []
    taste_means = []
    hearing_means = []
    smell_std = []
    taste_std = []
    hearing_std = []

    no_smell_means = []
    no_taste_means = []
    no_hearing_means = []
    no_smell_std = []
    no_taste_std = []
    no_hearing_std = []
    subjects = ['Subject 1', 'Subject 2', 'Subject 3', 'Subject 4', 'Subject 5']
    # subjects = []
    for idx, (subject, sdata) in enumerate(subject_predictions.items()):
        #     subjects.append(subject)
        for sense, ssdata in sdata.items():
            if 'Smell' in sense:
                smean = [ssdata['no_sense_mean'], ssdata['sense_mean']]
                error = [ssdata['no_sense_std'], ssdata['sense_std']]
                smell_means.append(smean[1])
                smell_std.append(error[1])
                no_smell_means.append(smean[0])
                no_smell_std.append(error[0])
            elif 'Hearing' in sense:
                smean = [ssdata['no_sense_mean'], ssdata['sense_mean']]
                error = [ssdata['no_sense_std'], ssdata['sense_std']]
                hearing_means.append(smean[1])
                hearing_std.append(error[1])
                no_hearing_means.append(smean[0])
                no_hearing_std.append(error[0])
            else:
                smean = [ssdata['no_sense_mean'], ssdata['sense_mean']]
                error = [ssdata['no_sense_std'], ssdata['sense_std']]
                taste_means.append(smean[1])
                taste_std.append(error[1])
                no_taste_means.append(smean[0])
                no_taste_std.append(error[0])

    # fig, ax = plt.subplots()
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    # fig.tight_layout()
    plt.subplots_adjust(hspace=0.4)

    width = 0.35  # the width of the bars
    ind = np.arange(len(subjects))  # the x locations for the groups

    axs[0].bar(ind, smell_means, width, yerr=smell_std, label='Sense')
    axs[0].bar(ind + width, no_smell_means, width, yerr=no_smell_std, label='No Sense')
    axs[0].set_title('ML predictions for Smell sense on all subjects')
    axs[0].set_xticks(ind + width / 2)
    axs[0].set_xticklabels(subjects, rotation=0)
    axs[0].legend(loc='lower right')
    # axs[0].yaxis.set_units(inch)
    axs[0].autoscale_view()

    axs[1].bar(ind, hearing_means, width, yerr=hearing_std, label='Sense')
    axs[1].bar(ind + width, no_hearing_means, width, yerr=no_hearing_std, label='No Sense')
    axs[1].set_title('ML predictions for Taste sense on all subjects')
    axs[1].set_xticks(ind + width / 2)
    axs[1].set_xticklabels(subjects, rotation=0)
    axs[1].legend(loc='lower right')
    # axs[1].yaxis.set_units(inch)
    axs[1].autoscale_view()

    axs[2].bar(ind, taste_means, width, yerr=taste_std, label='Sense')
    axs[2].bar(ind + width, no_taste_means, width, yerr=no_taste_std, label='No Sense')
    axs[2].set_title('ML predictions for Hearing sense on all subjects')
    axs[2].set_xticks(ind + width / 2)
    axs[2].set_xticklabels(subjects, rotation=0)
    axs[2].legend(loc='lower right')
    # axs[2].yaxis.set_units(inch)
    axs[2].autoscale_view()

    plt.savefig(fig_path)
    if to_show:
        plt.show()


def ml_pred_bar_plot_per_subject_per_sense(subject_predictions, fig_path='../models', to_show=False):
    action = ['non-active sense', 'active sense']
    x_pos = np.arange(len(action))
    for idx, (subject, sdata) in enumerate(subject_predictions.items()):
        #     idx += 2
        for sense, ssdata in sdata.items():
            smean = [ssdata['no_sense_mean'], ssdata['sense_mean']]
            error = [ssdata['no_sense_std'], ssdata['sense_std']]

            fig, ax = plt.subplots()
            ax.bar(x_pos, smean, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
            ax.set_ylabel("Model's predictions per sensory activity")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(action)
            ax.set_title("ML predictions for subject No.{} - {} sense".format(idx + 1, sense))
            ax.yaxis.grid(True)

            # Save the figure and show
            plt.tight_layout()
            plt.savefig('{/{}_{}_ml_pred_bar_plot_with_error_bars.png'.format(fig_path, subject, sense))
        if to_show:
            plt.show()


def ml_pred_bar_plot_per_sense(subject_predictions, fig_path='../models', to_show=False):
    senses_pred_data = dict()
    senses_pred_data['Smell'] = dict()
    senses_pred_data['Hearing'] = dict()
    senses_pred_data['Taste'] = dict()

    senses_pred_data['Smell']['no_sense_mean'] = list()
    senses_pred_data['Hearing']['no_sense_mean'] = list()
    senses_pred_data['Taste']['no_sense_mean'] = list()
    senses_pred_data['Smell']['no_sense_std'] = list()
    senses_pred_data['Hearing']['no_sense_std'] = list()
    senses_pred_data['Taste']['no_sense_std'] = list()

    senses_pred_data['Smell']['sense_mean'] = list()
    senses_pred_data['Hearing']['sense_mean'] = list()
    senses_pred_data['Taste']['sense_mean'] = list()
    senses_pred_data['Smell']['sense_std'] = list()
    senses_pred_data['Hearing']['sense_std'] = list()
    senses_pred_data['Taste']['sense_std'] = list()

    for sub, subdata in subject_predictions.items():
        for sens, sensdata in subdata.items():
            senses_pred_data[sens]['no_sense_mean'].append(sensdata['no_sense_mean'])
            senses_pred_data[sens]['sense_mean'].append(sensdata['sense_mean'])
            senses_pred_data[sens]['no_sense_std'].append(sensdata['no_sense_std'])
            senses_pred_data[sens]['sense_std'].append(sensdata['sense_std'])

    for sens, sensdata in senses_pred_data.items():
        senses_pred_data[sens]['smean'] = [np.mean(sensdata['no_sense_mean']), np.mean(sensdata['sense_mean'])]
        senses_pred_data[sens]['error'] = [np.mean(sensdata['no_sense_std']), np.mean(sensdata['sense_std'])]

    action = ['non-active sense', 'active sense']
    x_pos = np.arange(len(action))

    for k, v in senses_pred_data.items():
        fig, ax = plt.subplots()
        ax.bar(x_pos, v['smean'], yerr=v['error'], align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel("Model's predictions per sensory activity")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(action)
        ax.set_title("ML predictions for {} sense".format(k))
        ax.yaxis.grid(True)

        # Save the figure and show
        plt.tight_layout()
        plt.savefig('{}/{}_ml_pred_bar_plot_with_error_bars.png'.format(fig_path, k))
        if to_show:
            plt.show()
