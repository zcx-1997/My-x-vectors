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
import time

import numpy
import torch
import torchaudio

from torch.utils.data import Dataset, DataLoader
from python_speech_features import fbank, mfcc, delta

class Vox1_Train(Dataset):
    def __init__(self):
        self.random = random
        self.train_txt = r'data/vox1_train_100.txt'
        with open(self.train_txt, 'r') as f:
            self.wavs_labels = f.readlines()
        self.random.shuffle(self.wavs_labels)
        self.fixed_time = 5

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

        mfcc_f = mfcc(audio, sr, appendEnergy=True)
        delta_f = delta(mfcc_f, 2)
        feats = numpy.hstack((mfcc_f, delta_f))
        return feats

class Vox1_Test(Dataset):
    def __init__(self):
        self.random = random
        self.test_dir = r'/home/zcx/datasets/VoxCeleb/vox1_test_wav/wav'
        self.test_txt = r'data/veri_test100.txt'
        with open(self.test_txt, 'r') as f:
            self.test_pairs = f.readlines()
        # self.random.shuffle(self.test_pairs)
        self.fixed_time = 5

    def __getitem__(self, idx):
        label_pairs = self.test_pairs[idx].split()
        label = int(label_pairs[0])
        enroll_path = os.path.join(self.test_dir,label_pairs[1])
        test_path = os.path.join(self.test_dir,label_pairs[2])

        enroll_feats = self._load_data(enroll_path)
        enroll_feats = torch.tensor(enroll_feats, dtype=torch.float32)
        test_feats = self._load_data(test_path)
        test_feats = torch.tensor(test_feats, dtype=torch.float32)
        label = torch.tensor(label)

        return enroll_feats, test_feats, label


    def __len__(self):
        return len(self.test_pairs)

    def _load_data(self, wav_path):
        audio, sr = torchaudio.load(wav_path)
        audio = audio.reshape(-1)
        while len(audio) / sr < self.fixed_time:
            audio = torch.cat((audio, audio))
        time = len(audio) / sr
        audio = audio[:sr * self.fixed_time]
        time = len(audio) / sr  # 10s
        mfcc_f = mfcc(audio, sr, appendEnergy=True)
        delta_f = delta(mfcc_f, 2)
        feats = numpy.hstack((mfcc_f, delta_f))
        # feats = mfcc(audio, sr, numcep=25, appendEnergy=True)
        return feats


if __name__ == '__main__':
    print("======================== TRAIN ============================")
    train_db = Vox1_Train()
    print(len(train_db))
    x, y = train_db[0]
    print(x.shape, x.dtype)
    print(y, y.dtype)

    # print("单线程：")
    # print(time.ctime())
    # train_loader = DataLoader(train_db, batch_size=64, shuffle=True,
    #                           drop_last=True)
    #
    #
    # print(len(train_loader))
    # for i,(x,l) in enumerate(train_loader):
    #     if i < 10:
    #         pass
    #         print(time.ctime())
    #     else:
    #         break
    # print("done\n",time.ctime())

    # print("多线程：")
    # print(time.ctime())
    # train_loader = DataLoader(train_db, batch_size=64, shuffle=True,
    #                           drop_last=True,num_workers=6)
    #
    # print(len(train_loader))
    # for i,(x,l) in enumerate(train_loader):
    #     if i < 10:
    #         pass
    #         print(time.ctime())
    #     else:
    #         break
    # print("done\n", time.ctime())


    print("======================== TEST ============================")
    test_db = Vox1_Test()
    print(len(test_db))
    x, y, label = test_db[0]
    print(x.shape, x.dtype)
    print(y.shape, y.dtype)
    print(label, label.dtype)



    test_loader = DataLoader(test_db, batch_size=64, shuffle=True,
                              drop_last=True)
    print(len(test_loader))
    x, y, label = next(iter(test_loader))
    print(x.shape)
    print(y.shape)
    print(label.shape)

    for x, y, l in test_db:
        print(x.shape, x.dtype)
        print(y.shape, y.dtype)
        print(l, l.dtype)
        break