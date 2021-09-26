import os
import cv2
import glob
import shutil
import scipy.io
import scipy.misc
from tqdm import tqdm

# Data collection dates
TRAIN_DATES = ['20200114']
VAL_DATES = ['20200218']

# Data labels
SAMPLES = ['Zeev', 'Sergey', 'Yafim', 'Aviya']
SENSES = ['Smell', 'Hearing', 'Taste']
# Frames images size
FRAME_SIZE = 32

raw_data_path = '../raw_data'
data_path = '../data'


def split_video_to_frames(video_path, frames_path, frame_size):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        image = cv2.resize(image, (frame_size, frame_size))
        cv2.imwrite("{}/{}_frame_{}.jpg".format(frames_path, vid_filename.split(sep=os.sep)[-1], count), image)
        success, image = vidcap.read()
        #         print('Read a new frame: ', success)
        count += 1


def create_video_frames(dates, samples, senses):
    for date in dates:
        for sample in samples:
            for sense in senses:
                sense_videos_path = '{}/{}/{}/{}'.format(raw_data_path, date, sample, sense)
                no_sense_videos_path = '{}/{}/{}/No_{}'.format(raw_data_path, date, sample, sense)
                sense_frames_path = '{}/{}/{}/{}/frames'.format(data_path, 'train', sample, sense)
                no_sense_frames_path = '{}/{}/{}/No_{}/frames'.format(data_path, 'train', sample, sense)
                if not os.path.exists(sense_frames_path):
                    os.makedirs(sense_frames_path)
                if not os.path.exists(no_sense_frames_path):
                    os.makedirs(no_sense_frames_path)
                for vid_filename in tqdm(glob.glob('{}/*'.format(sense_videos_path))):
                    if os.path.isfile(vid_filename):
                        split_video_to_frames(vid_filename, sense_frames_path, FRAME_SIZE)
                for vid_filename in tqdm(glob.glob('{}/*'.format(no_sense_videos_path))):
                    if os.path.isfile(vid_filename):
                        split_video_to_frames(vid_filename, no_sense_frames_path, FRAME_SIZE)


def create_data_set(raw_data_path, data_path):
    # create training set
    create_video_frames(TRAIN_DATES, SAMPLES, SENSES)
    # create validation set
    create_video_frames(VAL_DATES, SAMPLES, SENSES)
