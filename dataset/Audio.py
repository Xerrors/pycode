import logging
from torch._C import dtype
from torch.utils.data import DataLoader, Dataset
import os
import zipfile
import librosa
import pandas as pd
import numpy as np
import torch

import time
import torchaudio


data_dir = "./data"


def mel(wav_path):
    waveform, sample_rate = torchaudio.load(wav_path)
    S = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            win_length=1024,
            hop_length=500,
            f_min=50,
            f_max=14000,
            n_mels=64,
            n_fft=2048)(waveform)
    S = S.reshape(1, 64, -1)
    S = S[:, :, :400]
    return S


def mel_fun(wav_path):
    waveform, sample_rate = librosa.load(wav_path, sr=None)
    print("SR:", sample_rate)

    INPUT_LEN = sample_rate * 10   # 5秒长度
    WAVE_LEN = len(waveform)

    # 音频过长，裁剪，音频过短，填充
    if WAVE_LEN < INPUT_LEN:
        waveform = np.concatenate((x, np.zeros(max_len - len(x))))
    elif WAVE_LEN > INPUT_LEN:
        waveform = waveform[0 : INPUT_LEN]

    mel = librosa.feature.melspectrogram(waveform, sr=sample_rate, n_fft=2048, n_mels=64) 
    mel = librosa.power_to_db(mel).T

    mel = mel.reshape(1, mel.shape[0], -1)

    return mel


class TAUUrbanAcousticScenes(Dataset):
    """ TAU 数据集说明 """

    def __init__(self, root: str, train: bool = True, transform = None, unzip: bool = False):
        self.dataset_path = root
        self.meta_file = os.path.join(self.dataset_path, 'meta.csv')
        self.train_file = os.path.join(self.dataset_path, 'evaluation_setup', 'fold1_train.csv')
        self.test_file = os.path.join(self.dataset_path, 'evaluation_setup', 'fold1_evaluate.csv')
        self.labels = ['indoor', 'outdoor', 'transportation']

        if unzip or not os.path.exists(os.path.join(self.dataset_path, 'meta.csv')):
            zip_files = [
                "TAU-urban-acoustic-scenes-2020-3class-development.audio.1.zip",
                "TAU-urban-acoustic-scenes-2020-3class-development.audio.2.zip",
                "TAU-urban-acoustic-scenes-2020-3class-development.audio.3.zip",
                "TAU-urban-acoustic-scenes-2020-3class-development.audio.4.zip",
                "TAU-urban-acoustic-scenes-2020-3class-development.audio.5.zip",
                "TAU-urban-acoustic-scenes-2020-3class-development.audio.6.zip",
                "TAU-urban-acoustic-scenes-2020-3class-development.audio.7.zip",
                "TAU-urban-acoustic-scenes-2020-3class-development.audio.8.zip",
                "TAU-urban-acoustic-scenes-2020-3class-development.audio.9.zip",
                "TAU-urban-acoustic-scenes-2020-3class-development.audio.10.zip",
                "TAU-urban-acoustic-scenes-2020-3class-development.audio.11.zip",
                "TAU-urban-acoustic-scenes-2020-3class-development.audio.12.zip",
                "TAU-urban-acoustic-scenes-2020-3class-development.audio.13.zip",
                "TAU-urban-acoustic-scenes-2020-3class-development.audio.14.zip",
                "TAU-urban-acoustic-scenes-2020-3class-development.audio.15.zip",
                "TAU-urban-acoustic-scenes-2020-3class-development.audio.16.zip",
                "TAU-urban-acoustic-scenes-2020-3class-development.audio.17.zip",
                "TAU-urban-acoustic-scenes-2020-3class-development.audio.18.zip",
                "TAU-urban-acoustic-scenes-2020-3class-development.audio.19.zip",
                "TAU-urban-acoustic-scenes-2020-3class-development.audio.20.zip",
                "TAU-urban-acoustic-scenes-2020-3class-development.audio.21.zip",
                "TAU-urban-acoustic-scenes-2020-3class-development.doc.zip",
                "TAU-urban-acoustic-scenes-2020-3class-development.meta.zip",
            ]
            soruce_data_dir = '/data1/dcase'
            for zip_file in zip_files:
                zip_file_path = os.path.join(soruce_data_dir, zip_file)
                assert os.path.exists(zip_file_path)
                print('Extracting file: ', zip_file_path)
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                # os.remove(zip_file_path)  # delete zipped file
            print('Done!')

        if train:
            csv_filename = self.train_file
        else:
            csv_filename = self.test_file

        self.file_lists = pd.read_csv(csv_filename, delimiter='\t')
         

    def __getitem__(self, index):
        audio, label = self.file_lists.iloc[index]
        audio = os.path.join(self.dataset_path, audio)
        audio = mel_fun(audio)
        label = self.labels.index(label)
        return audio, label

    def __len__(self):
        return len(self.file_lists)


def TAUUrbanAcousticScenes2020_3classDevelopment(batch_size=128, num_worker=8, unzip=False):
    """ TAU Urban Acoustic Scenes 2020 3Class
    - Development dataset (41.5G)<https://doi.org/10.5281/zenodo.3670185>
    - Evaluation dataset (19.9 GB)<https://doi.org/10.5281/zenodo.3685835>
    """
    ds_dir = 'TAU-urban-acoustic-scenes-2020-3class-development'
    train_dataset = TAUUrbanAcousticScenes(os.path.join(data_dir, ds_dir), train=True, unzip=unzip)
    train_loader = DataLoader(train_dataset, drop_last=True, batch_size=batch_size, num_workers=num_worker, pin_memory=True)

    test_dataset = TAUUrbanAcousticScenes(os.path.join(data_dir, ds_dir), train=False, unzip=unzip)
    test_loader = DataLoader(test_dataset, drop_last=True, batch_size=batch_size, num_workers=num_worker, pin_memory=True)

    return train_loader, test_loader


""" 下面的代码是参考别人的数据集 """

import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
# from feature import calculate_feature_for_all_audio_files
import numpy as np
import pandas as pd
import pickle
import os
# import config



def read_pkl(path):

    f=open(path,'rb')
    data = pickle.load(f)
    return data

def data_generate():
    ds_dir = os.path.join(data_dir, 'TAU-urban-acoustic-scenes-2020-3class-development')

    train_data = read_pkl(os.path.join(ds_dir, './Train_feature_dict.pkl'))
    test_data = read_pkl(os.path.join(ds_dir, './Test_feature_dict.pkl'))

    x_train = train_data['feature']
    y_train = train_data['label']
    x_test = test_data['feature']
    y_test = test_data['label']
    y_train = np.array(pd.get_dummies(y_train))
    y_test = np.array(pd.get_dummies(y_test))

    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    x_train, y_train, x_test, y_test = map(torch.tensor, (x_train, y_train, x_test, y_test))

    x_train = torch.unsqueeze(x_train, dim=1).float()
    x_test = torch.unsqueeze(x_test, dim=1).float()

    return x_train,y_train,x_test, y_test


def TAUUrbanAcousticScenes2020_3classDevelopment_Feature(batch_size=128, num_worker=8):
    """ 提取了特征之后的数据集 """
    ds_dir = 'TAU-urban-acoustic-scenes-2020-3class-development'
    x_train,y_train,x_test, y_test= data_generate()
    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_worker),
        DataLoader(test_ds, batch_size=batch_size * 2, num_workers=num_worker),
    )


