import os
import time

import numpy as np
import torch
from scipy import signal
from scipy.signal import butter, lfilter
from torch.utils.data import Dataset


def rms_pooling(emg, window_size=3, stride=2):
    L1 = emg.shape[0]
    L2 = (L1 - window_size) // stride + 1
    new_emg = np.zeros((L2, emg.shape[1]))
    for i in range(L2):
        new_emg[i, :] = np.sqrt(np.mean(emg[i * stride:i * stride + window_size, :] ** 2, axis=0))
    return new_emg


def get_data_ninapro_db1(path_s, seq_lens):
    seq_len = seq_lens
    step = 1
    emgs = np.loadtxt(os.path.join(path_s, 'emg.txt'))
    labels = np.loadtxt(os.path.join(path_s, 'restimulus.txt'))
    repetitions = np.loadtxt(os.path.join(path_s, 'rerepetition.txt'))
    exercise = np.loadtxt(os.path.join(path_s, 'exercise.txt'))

    # perform 1-order 1Hz low-pass filter
    order = 1
    fs = 100  # sample rate: 100Hz
    cutoff = 1  # cutoff frequency
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, 'lowpass')
    for _col in range(emgs.shape[1]):
        emgs[:, _col] = signal.filtfilt(b, a, emgs[:, _col])

    # u-law normalization
    u = 256
    emgs = np.sign(emgs) * np.log(1 + u * abs(emgs)) / np.log(1 + u)

    # segmentation of training and testing samples
    length_dots = len(labels)
    data_train = []
    labels_train = []
    exercise_train = []
    data_val = []
    labels_val = []
    exercise_val = []
    for idx in range(0, length_dots - length_dots % seq_len, step):
        if labels[idx] > 0 and labels[idx + seq_len - 1] > 0 and labels[idx] == labels[idx + seq_len - 1]:
            repetition = repetitions[idx]
            if repetition in [2, 5, 7]:  # val dataset
                data_val.append(emgs[idx:idx + seq_len, :])
                labels_val.append(labels[idx])
                exercise_val.append(exercise[idx])
            else:  # train dataset
                data_train.append(emgs[idx:idx + seq_len, :])
                labels_train.append(labels[idx])
                exercise_train.append(exercise[idx])
    return data_train, labels_train, exercise_train, data_val, labels_val, exercise_val


def get_data_ninapro_db2(path_s, seq_lens):
    seq_len = seq_lens
    step_train = 40  # stride = 20ms for 2000Hz,
    step_val = 200  # stride = 100ms for 2000Hz,

    emgs = np.loadtxt(os.path.join(path_s, 'emg.txt'))
    labels = np.loadtxt(os.path.join(path_s, 'restimulus.txt'))
    repetitions = np.loadtxt(os.path.join(path_s, 'rerepetition.txt'))
    exercise = np.loadtxt(os.path.join(path_s, 'exercise.txt'))

    # perform 4-order 150Hz low-pass filter
    fs = 2000  # Sample rate
    cutoff = 150  # cut-off frequency
    order = 4  #
    # Butterworth filter
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    for c in range(12):
        emgs[:, c] = lfilter(b, a, emgs[:, c])

    # down-sampling 2000Hz-->200Hz
    # factor = 5
    # emgs = emgs[::factor, :]
    # labels = labels[::factor]
    # repetitions = repetitions[::factor]
    # exercise = exercise[::factor]
    # seq_len = seq_len // factor
    # step_train = step_train // factor
    # step_val = step_val // factor

    # min-max normalization
    _norm = max(abs(emgs.max()), abs(emgs.min()))
    emgs = emgs / _norm

    # scale = 0.1/np.mean(np.abs(emgs))
    # emgs = emgs * scale

    # u-law normalization
    u = 256
    emgs = np.sign(emgs) * np.log(1 + u * abs(emgs)) / np.log(1 + u)

    # segmentation of training and testing samples
    length_dots = len(labels)
    data_train = []
    labels_train = []
    exercise_train = []
    data_val = []
    labels_val = []
    exercise_val = []
    step = step_val
    for idx in range(0, length_dots - length_dots % seq_len, step):
        # step = 4 * step_val if step == step_val else step_val
        try:
            if labels[idx] > 0 and labels[idx + seq_len - 1] > 0 and labels[idx] == labels[idx + seq_len - 1]:
                repetition = repetitions[idx]
                if repetition in [2, 5]:  # val dataset
                    sample = rms_pooling(emgs[idx:idx + seq_len, :])
                    data_val.append(sample)
                    if 1 <= labels[idx] <= 17:
                        labels_val.append(labels[idx])
                    elif 35 <= labels[idx] <= 57:
                        labels_val.append(labels[idx] - 17)
                    elif 98 <= labels[idx] <= 106:
                        labels_val.append(labels[idx] - 57)
                    else:
                        print('get labels wrong!')
                        time.sleep(1000000)
                    exercise_val.append(exercise[idx])
        except:
            pass
    step = step_train
    for idx in range(0, length_dots - length_dots % seq_len, step):
        # step = 4 * step_train if step == step_train else step_train
        try:
            if labels[idx] > 0 and labels[idx + seq_len - 1] > 0 and labels[idx] == labels[idx + seq_len - 1]:
                repetition = repetitions[idx]
                if repetition in [1, 3, 4, 6]:  # train dataset [1,3,4,6]
                    sample = rms_pooling(emgs[idx:idx + seq_len, :])
                    data_train.append(sample)
                    if 1 <= labels[idx] <= 17:
                        labels_train.append(labels[idx])
                    elif 35 <= labels[idx] <= 57:
                        labels_train.append(labels[idx] - 17)
                    elif 98 <= labels[idx] <= 106:
                        labels_train.append(labels[idx] - 57)
                    else:
                        print('get labels wrong!')
                        time.sleep(1000000)
                    exercise_train.append(exercise[idx])
        except:
            pass
    return data_train, labels_train, exercise_train, data_val, labels_val, exercise_val


class Ninapro(Dataset):
    def __init__(self, emgs, labels) -> None:
        super().__init__()
        self.emgs = emgs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = self.emgs[index]
        sample = torch.tensor(sample).permute(1, 0).unsqueeze(0).float().to('cuda')
        label = self.labels[index] - 1
        label = torch.tensor(label).long().to('cuda')
        return sample, label


class Ninapro_db1(Dataset):
    def __init__(self, emgs, labels, exercise) -> None:
        super().__init__()
        self.emgs = emgs
        self.labels = labels
        self.exercise = exercise

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = self.emgs[index]
        sample = torch.tensor(sample).permute(1, 0).unsqueeze(0).float().to('cuda')
        label = self.labels[index] + 2  # 3~55
        # label = torch.tensor(label).long().to('cuda')
        exercise = self.exercise[index] - 1  # 0~2
        # exercise = torch.tensor(exercise).long().to('cuda')
        return sample, label, exercise


class Ninapro_db2(Dataset):
    def __init__(self, emgs, labels, exercise) -> None:
        super().__init__()
        self.emgs = emgs
        self.labels = labels
        self.exercise = exercise

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = self.emgs[index]
        sample = torch.tensor(sample).permute(1, 0).unsqueeze(0).float().to('cuda')
        label = self.labels[index] + 2  # 3~52
        # label = torch.tensor(label).long().to('cuda')
        exercise = self.exercise[index] - 1  # 0~2
        # exercise = torch.tensor(exercise).long().to('cuda')
        return sample, label, exercise


def collate_fn(batch_data):
    def creat_attn_mask(L):
        mask = torch.arange(L).unsqueeze(-1) < torch.arange(L).unsqueeze(0).bool()
        return mask

    src = []
    tgt = []
    labels = []
    for x, label, exercise in batch_data:
        src.append(x.tolist())
        tgt.append([0, exercise + 1])  # [<BOS>,exercise]  <BOS>(0) exercise(1,2,3)
        labels.append([exercise, label])  # [exercise,gesture] exercise(0,1,2) gesture(3,4,5,...)
    tgt_mask = creat_attn_mask(2).to('cuda')

    src = torch.tensor(src).float().to('cuda')
    tgt = torch.tensor(tgt).long().to('cuda')
    labels = torch.tensor(labels).long().to('cuda')

    return src, tgt, tgt_mask, labels
