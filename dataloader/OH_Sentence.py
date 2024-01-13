import os

import numpy as np
import torch
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cut_emg(data, h):
    start = 0
    end = len(data) - 1
    lens = len(data)
    window_size = 20
    for i in range(lens - window_size + 1):
        g = (abs(data[i:i + window_size]) > h)
        if g.sum() > window_size / 2:
            start = i
            break
    for i in range(lens - window_size + 1):
        g = (abs(data[lens - i - 1 - window_size:lens - i - 1]) > h)
        if g.sum() > window_size / 2:
            end = lens - i - 1
            break
    return start, end


def cut_imu(data, h):
    start = 0
    end = len(data) - 1
    lens = len(data)
    window_size = 40
    for i in range(lens - window_size + 1):
        g = (abs(data[i:i + window_size])) > h
        if g.sum() > window_size / 2:
            start = i
            break
    for i in range(lens - window_size + 1):
        g = (abs(data[lens - i - 1 - window_size:lens - i - 1]) > h)
        if g.sum() > window_size / 2:
            end = lens - i - 1
            break
    return start * 4, end * 4


def signal_filter(emg):
    fs = 200
    cutoff = 30

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    order = 4

    b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
    channels = emg.shape[1]
    for channel in range(channels):
        emg[:, channel] = filtfilt(b, a, emg[:, channel])
    return emg


def rms_pooling(emg, window_size=2, stride=2):
    L1 = emg.shape[0]
    L2 = (L1 - window_size) // stride + 1
    new_emg = np.zeros((L2, emg.shape[1]))
    for i in range(L2):
        new_emg[i, :] = np.sqrt(np.mean(emg[i * stride:i * stride + window_size, :] ** 2, axis=0))
    return new_emg


def u_law_normalization(emg, u=255):
    emg = np.sign(emg) * np.log(1 + u * abs(emg)) / np.log(1 + u)
    return emg


def z_score_normalization(x, mu, sigma=1):
    x = x.astype(float)
    if sigma != 0:
        x = (x - mu) / sigma
    else:
        x = (x - mu) / sigma
    return x


def pdm_normalization(x):
    return x / x.max(axis=0)


def signal_split(emg, window_size, stride):
    splited_emg = []
    emg_lens = emg.shape[0]
    N = (emg_lens - window_size) // stride + 1
    for i in range(N):
        splited_emg.append(emg[i * stride:i * stride + window_size, :])
    return splited_emg


def emg_preprocessing(emg_path, emg_file, window_size, stride):
    emg_path = os.path.join(emg_path, emg_file)
    emg = np.loadtxt(emg_path)
    imu_path = emg_path.replace('emg', 'imu')
    imu = np.loadtxt(imu_path)
    start1, end1 = cut_emg(np.mean(emg, axis=1), 2)
    start2, end2 = cut_imu(imu[:, 9], 5)
    start = max(start1, start2)
    end = min(end1, end2)
    emg = emg[start:end, :]
    # emg = signal_filter(emg)
    # emg = rms_pooling(emg)
    # emg=pdm_normalization(emg)
    # min-max normalization
    _norm = max(abs(emg.max()), abs(emg.min()))
    emg = emg / _norm
    # emg = u_law_normalization(emg)

    # print(emg.shape[0])

    splited_emg = signal_split(emg, window_size, stride)
    return splited_emg


def make_data_oh(path, window_size, stride):
    label_encoder = LabelEncoder()
    emg_path = os.path.join(path, 'emg')
    all_emg_files = os.listdir(emg_path)

    train_data = []
    train_labels = []

    test_data = []
    test_labels = []

    for emg_file in all_emg_files:
        seq = []
        try:
            repetition = emg_file.split('_')[2]
            if repetition not in ['3', '7']:
                emg_preprocessed = emg_preprocessing(emg_path, emg_file, window_size, stride)  #
                L = len(emg_preprocessed)
                emg_preprocessed = np.array(emg_preprocessed)  #
                train_data.append(emg_preprocessed)
                train_labels.append(emg_file.split('_')[0])
            else:
                emg_preprocessed = emg_preprocessing(emg_path, emg_file, window_size, stride)  #
                L = len(emg_preprocessed)
                emg_preprocessed = np.array(emg_preprocessed)  #
                test_data.append(emg_preprocessed)
                test_labels.append(emg_file.split('_')[0])
        except:
            pass
    return train_data, label_encoder.fit_transform(train_labels), test_data, label_encoder.transform(test_labels)


class OH_Dataset(Dataset):
    def __init__(self, data, labels):
        super(OH_Dataset, self).__init__()
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]).float(), torch.tensor(self.labels[idx]).long()

    def __len__(self):
        assert len(self.data) == len(self.labels)
        return len(self.labels)


def creat_padding_mask(L, seq_lens):
    if type(seq_lens) == list:
        seq_lens = torch.tensor(seq_lens)
    seq_lens = seq_lens.unsqueeze(-1)  # Bx1
    mask = torch.arange(L) < seq_lens  # BxL
    mask = mask.unsqueeze(-1)  # BxLx1
    return ~mask


def collate_fn1(batch_data):
    seqs = []
    labels = []
    seq_lens = []
    max_L = 0
    for x, y in batch_data:
        try:
            if x.shape[0] < 50:  #
                seqs.append(x.transpose(1, 2))  #
                labels.append(y)
                seq_lens.append(x.shape[0])
                max_L = max(max_L, x.shape[0])
            else:
                pass
        except:
            pass

    src = pad_sequence(seqs, batch_first=True)
    padding_mask = creat_padding_mask(max_L, seq_lens)

    return src.float().to(device), torch.tensor(labels).long().to(device), torch.tensor(seq_lens).long().to(
        device), padding_mask.to(device)


def creat_key_padding_mask(L, seq_lens):
    if type(seq_lens) == list:
        seq_lens = torch.tensor(seq_lens)
    seq_lens = seq_lens.unsqueeze(-1)  # Bx1
    mask = torch.arange(L) < seq_lens  # BxL
    return ~mask


def collate_fn2(batch_data):
    seqs = []
    labels = []
    seq_lens = []
    max_L = 0
    for x, y in batch_data:
        try:
            seqs.append(x.transpose(1, 2))  #
            labels.append(y)
            seq_lens.append(x.shape[0])
            max_L = max(max_L, x.shape[0])
        except:
            pass

    src = pad_sequence(seqs, batch_first=True)
    src_key_padding_mask = creat_key_padding_mask(max_L, seq_lens)

    return src.float().to(device), torch.tensor(labels).long().to(device), src_key_padding_mask.to(device)


'''
exp5:Transformer&LSTM
'''

from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence

label_encoder = LabelEncoder()


def seqmodels_signal_split(emg, window_size, stride):
    splited_emg = []
    emg_lens = emg.shape[0]
    N = (emg_lens - window_size) // stride + 1
    for i in range(N):
        splited_emg.append(emg[i * stride:i * stride + window_size, :].reshape(-1))  #
    return splited_emg


def seqmodels_emg_preprocessing(emg_path, emg_file, window_size, stride):
    emg_path = os.path.join(emg_path, emg_file)
    emg = np.loadtxt(emg_path)
    imu_path = emg_path.replace('emg', 'imu')
    imu = np.loadtxt(imu_path)

    start1, end1 = cut_emg(np.mean(emg, axis=1), 2)
    start2, end2 = cut_imu(imu[:, 9], 5)
    start = max(start1, start2)
    end = min(end1, end2)
    emg = emg[start:end, :]
    # emg = signal_filter(emg)

    # emg = rms_pooling(emg)
    # emg=pdm_normalization(emg)
    # min-max normalization
    _norm = max(abs(emg.max()), abs(emg.min()))
    emg = emg / _norm
    # emg = u_law_normalization(emg)
    # print(emg.shape[0])

    splited_emg = seqmodels_signal_split(emg, window_size, stride)
    return splited_emg


def seqmodels_make_data(path, window_size, stride):
    emg_path = os.path.join(path, 'emg')
    all_emg_files = os.listdir(emg_path)
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for emg_file in all_emg_files:
        seq = []
        try:
            repetition = emg_file.split('_')[2]
            if repetition not in ['3', '7']:
                emg_preprocessed = seqmodels_emg_preprocessing(emg_path, emg_file, window_size,
                                                               stride)  # list of segment
                L = len(emg_preprocessed)
                emg_preprocessed = np.array(emg_preprocessed)  #
                train_data.append(emg_preprocessed)
                train_labels.append(emg_file.split('_')[0])
            else:
                emg_preprocessed = seqmodels_emg_preprocessing(emg_path, emg_file, window_size, stride)
                L = len(emg_preprocessed)
                emg_preprocessed = np.array(emg_preprocessed)  #
                test_data.append(emg_preprocessed)
                test_labels.append(emg_file.split('_')[0])
        except:
            pass
    return train_data, label_encoder.fit_transform(train_labels), test_data, label_encoder.transform(test_labels)


class seqmodels_Dataset(Dataset):
    def __init__(self, data, labels):
        super(seqmodels_Dataset, self).__init__()
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]).float(), torch.tensor(self.labels[idx]).long()

    def __len__(self):
        assert len(self.data) == len(self.labels)
        return len(self.labels)


def seqmodels_creat_padding_mask(L, seq_lens):
    if type(seq_lens) == list:
        seq_lens = torch.tensor(seq_lens)
    seq_lens = seq_lens.unsqueeze(-1)  # Bx1
    mask = torch.arange(L) < seq_lens  # BxL
    mask = mask.unsqueeze(-1)  # BxLx1
    return ~mask


def seqmodels_collate_fn1(batch_data):
    seqs = []
    labels = []
    seq_lens = []
    max_L = 0
    for x, y in batch_data:
        try:
            if x.shape[-1] == 150 * 8:
                seqs.append(x)
                labels.append(y)
                seq_lens.append(x.shape[0])
                max_L = max(max_L, x.shape[0])
            else:
                pass
        except:
            pass

    src = pad_sequence(seqs, batch_first=True)
    padding_mask = seqmodels_creat_padding_mask(max_L, seq_lens)

    return src.float().to(device), torch.tensor(labels).long().to(device), torch.tensor(seq_lens).long().to(
        device), padding_mask.to(device)


def seqmodels_creat_key_padding_mask(L, seq_lens):
    if type(seq_lens) == list:
        seq_lens = torch.tensor(seq_lens)
    seq_lens = seq_lens.unsqueeze(-1)  # Bx1
    mask = torch.arange(L) < seq_lens  # BxL
    return ~mask


def seqmodels_collate_fn2(batch_data):
    seqs = []
    labels = []
    seq_lens = []
    for x, y in batch_data:
        try:
            if x.shape[-1] == 150 * 8:
                seqs.append(x)
                labels.append(y)
                seq_lens.append(x.shape[0])
        except:
            pass
    max_L = max([len(seq) for seq in seqs])
    src = pad_sequence(seqs, batch_first=True)
    mask = seqmodels_creat_key_padding_mask(max_L, seq_lens)

    return src.float().to(device), torch.tensor(labels).long().to(device), mask.to(device)
