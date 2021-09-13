# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My-X-vector -> data_loader
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/9/13 下午3:16
@Description        ：
                    _ooOoo_    
                   o8888888o    
                   88" . "88    
                   (| -_- |)    
                    O\ = /O    
                ____/`---'\____    
                 .' \\| |// `.    
               / \\||| : |||// \    
             / _||||| -:- |||||- \    
               | | \\\ - /// | |    
             | \_| ''\---/'' | |    
              \ .-\__ `-` ___/-. /    
           ___`. .' /--.--\ `. . __    
        ."" '< `.___\_<|>_/___.' >'"".    
       | | : `- \`.;`\ _ /`;.`/ - ` : | |    
         \ \ `-. \_ __\ /__ _/ .-` / /    
 ======`-.____`-.___\_____/___.-`____.-'======    
                    `=---='    
 .............................................    
              佛祖保佑             永无BUG
==================================================
"""
import os
import glob
import random

import torch
import torchaudio

from torch.utils.data import Dataset, DataLoader
from python_speech_features import fbank, mfcc

train_dir = r'/home/zcx/datasets/VoxCeleb/vox1_dev_wav/wav'


class Vox1_Train(Dataset):
    def __init__(self):
        self.random = random
        self.train_txt = r'data/vox1_train.txt'
        with open(self.train_txt, 'r') as f:
            self.wavs_labels = f.readlines()
        self.random.shuffle(self.wavs_labels)
        self.fixed_time = 10

    def __getitem__(self, idx):
        wav_label = self.wavs_labels[idx].split()
        wav_path = wav_label[0]
        label = int(wav_label[1])
        feats = self._load_data(wav_path)
        feats = torch.tensor(feats, dtype=torch.float32)
        label = torch.tensor(label)
        return feats, label

    def __len__(self):
        return len(self.wavs_labels)

    def _load_data(self, wav_path):
        audio, sr = torchaudio.load(wav_path)
        audio = audio.reshape(-1)
        while len(audio) / sr < self.fixed_time:
            audio = torch.cat((audio, audio))
        time = len(audio) / sr
        audio = audio[:sr * self.fixed_time]
        time = len(audio) / sr  # 10s
        feats = mfcc(audio, sr, numcep=20, appendEnergy=True)
        return feats


if __name__ == '__main__':
    train_db = Vox1_Train()
    print(len(train_db))
    x, y = train_db[0]
    print(x.shape, x.dtype)
    print(y, y.dtype)

    train_loader = DataLoader(train_db, batch_size=64, shuffle=True,
                              drop_last=True)
    x, y = next(iter(train_loader))
    print(x.shape)
    print(y.shape)
