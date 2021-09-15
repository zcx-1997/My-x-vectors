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




