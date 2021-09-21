# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My-X-vector -> test
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/9/13 下午4:20
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
import random
import torch

a = torch.arange(2)
b = torch.arange(3)
print(a,b)
c = torch.cat((a,b))
print(c)

filename = r'data/vox1_train.txt'
with open(filename,'r') as f:
    wavs_labels = f.readlines()
    print(len(wavs_labels))
    # print(path_labels)
    print(wavs_labels[0])
    random.shuffle(wavs_labels)
    print(wavs_labels[0])


a = torch.arange(12).view(3,4).float()
b = torch.arange(1,13).view(3,4).float()
cossim = torch.cosine_similarity(a,b)
print(cossim)

print("============TEST DATASET================")
from data_loader import Vox1_Test

test_db = Vox1_Test()

x = torch.rand(2,3,4)
net = torch.nn.Linear(4,1)
y = net(x)
print(y.shape)

start_epoch = 5
for epoch in range(start_epoch,10):
    print(epoch)

import time
import torch.utils.data as d
import torchvision
import torchvision.transforms as transforms

if __name__ == '__main__':
    BATCH_SIZE = 100
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.MNIST('\mnist', download=True, train=True,
                                           transform=transform)

    # data loaders
    train_loader = d.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    for num_workers in range(20):import time
import torch.utils.data as d
import torchvision
import torchvision.transforms as transforms


if __name__ == '__main__':
    BATCH_SIZE = 100
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.MNIST('\mnist', download=False, train=True, transform=transform)

    # data loaders
    train_loader = d.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    for num_workers in range(20):
        train_loader = d.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
        # training ...
        start = time.time()
        for epoch in range(1):
            for step, (batch_x, batch_y) in enumerate(train_loader):
                pass
        end = time.time()
        print('num_workers is {} and it took {} seconds'.format(num_workers, end - start))
        train_loader = d.DataLoader(train_set, batch_size=BATCH_SIZE,
                                    shuffle=True, num_workers=num_workers)
        # training ...
        start = time.time()
        for epoch in range(1):
            for step, (batch_x, batch_y) in enumerate(train_loader):
                pass
        end = time.time()
        print('num_workers is {} and it took {} seconds'.format(num_workers,
                                                                end - start))




