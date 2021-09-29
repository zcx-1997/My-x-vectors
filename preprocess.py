# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My-X-vector -> preprocess
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/9/13 下午4:32
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

train_dir = r'/home/zcx/datasets/VoxCeleb/vox1_dev_wav/wav'
spks_list = os.listdir(train_dir)
# random.shuffle(spks_list)
# spks_list = spks_list[:500]

with open('data/vox1_train.txt','w') as f:
    for spk_name in spks_list:
        spk_id = int(spk_name[-4:])
        wavs_list = glob.glob(os.path.join(train_dir, spk_name, '*/*'))
        for wav_path in wavs_list:
            # print(wav_path, spk_id)
            f.write(wav_path+' '+str(spk_id)+'\n')


