# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My-X-vector -> model
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/9/13 下午8:08
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
import torch
from torch import nn

from torchsummary import summary

import constants as c


class TDNN(nn.Module):
    def __init__(self):
        super(TDNN, self).__init__()
        self.tdnn1 = nn.Conv1d(in_channels=26, out_channels=512, kernel_size=5,
                               dilation=1)
        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,
                               dilation=2)
        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,
                               dilation=2)
        self.tdnn4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1,
                               dilation=1)
        self.tdnn5 = nn.Conv1d(in_channels=512, out_channels=1500,
                               kernel_size=1, dilation=1)

        self.embedding7 = nn.Linear(3000, 512)
        self.fc8 = nn.Linear(512, 512)
        self.fc9 = nn.Linear(512, 1252)

        # self.bn1 = nn.BatchNorm1d(128, momentum=0.1, affine=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1500)

        # self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batch,embedding_size,num_frames)
        # output: (batch, embeddings)
        x = self.bn1(self.relu(self.tdnn1(x)))
        x = self.bn2(self.relu(self.tdnn2(x)))
        x = self.bn3(self.relu(self.tdnn3(x)))
        x = self.bn4(self.relu(self.tdnn4(x)))
        x = self.bn5(self.relu(self.tdnn5(x))).transpose(1, 2)

        x_stat = torch.cat((x.mean(dim=1), x.std(dim=1)), dim=1)

        embedding = self.embedding7(x_stat)
        x = self.relu(embedding)
        x = self.relu(self.fc8(x))
        x = self.fc9(x)
        return x, embedding


class My_E_TDNN(nn.Module):
    def __init__(self):
        super(My_E_TDNN, self).__init__()
        self.tdnn1 = nn.Conv1d(in_channels=25, out_channels=512, kernel_size=5,
                               dilation=1)
        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,
                               dilation=2)
        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,
                               dilation=3)
        self.tdnn4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,
                               dilation=4)
        # self.tdnn5 = nn.Conv1d(in_channels=512, out_channels=1500,
        #                        kernel_size=1, dilation=1)

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)

        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 1500)


        self.embedding = nn.Linear(3000, 512)
        self.fc8 = nn.Linear(512, 512)
        self.fc9 = nn.Linear(512, 1252)

        # self.bn1 = nn.BatchNorm1d(128, momentum=0.1, affine=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(512)

        # self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batch,embedding_size,num_frames)
        # output: (batch, embeddings)
        x = self.bn1(self.relu(self.tdnn1(x))).transpose(1, 2)
        x = self.relu(self.fc1(x)).transpose(1, 2)
        x = self.bn2(self.relu(self.tdnn2(x))).transpose(1, 2)
        x = self.relu(self.fc2(x)).transpose(1, 2)
        x = self.bn3(self.relu(self.tdnn3(x))).transpose(1, 2)
        x = self.relu(self.fc3(x)).transpose(1, 2)
        x = self.bn4(self.relu(self.tdnn4(x))).transpose(1, 2)
        x = self.relu(self.fc4(x))

        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))

        x_stat = torch.cat((x.mean(dim=1), x.std(dim=1)), dim=1)

        embedding = self.embedding(x_stat)
        x = self.relu(embedding)
        x = self.relu(self.fc8(x))
        x = self.fc9(x)
        return x, embedding


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = TDNN().to(device)
    summary(net, input_size=(26, 299))
