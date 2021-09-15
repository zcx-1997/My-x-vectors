# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My-X-vector -> train
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/9/13 下午8:24
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
import time

import torch
import random
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import constants as c
from data_loader import Vox1_Train, Vox1_Test

from model import MyTDNN


# 计算准确率
def sum_right(y_hat, y):
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y  # 结果是一个包含0（错）和1（对）的张量
    return float(cmp.type(y.dtype).sum())


def accuracy(y_hat, y):
    """计算准确率。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y  # 结果是一个包含0（错）和1（对）的张量
    return float(cmp.type(y.dtype).sum()) / len(y)


def train(device):
    checkpoint_dir = r'checkpoints'
    log_file = os.path.join(checkpoint_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_db = Vox1_Train()
    train_loader = DataLoader(train_db, batch_size=64, shuffle=True,
                              drop_last=True)
    net = MyTDNN()
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(net.parameters(), 0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(opt,step_size=50,gamma=0.5)
    writer = SummaryWriter()

    print("start train ...")
    for epoch in range(10):
        net.train()
        total_loss, epoch_rights, all_sample = 0, 0, 0
        step_rights = 0
        for step_id, (x, y) in enumerate(train_loader):
            x = x.transpose(1, 2).to(device)
            y = y.to(device)
            y_hat, _, _ = net(x)
            loss = criterion(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # scheduler.step()

            writer.add_scalar("loss", loss, epoch * step_id)

            step_rights = sum_right(y_hat, y)

            total_loss += loss
            epoch_rights += sum_right(y_hat, y)
            all_sample += len(y)

            print("Epoch{},Step[{}/{}]: loss={:.4f},acc={:.4f}, time={}".format(epoch + 1,
                step_id + 1, len(train_loader), loss, step_rights / len(y),time.ctime()))

        avg_loss = total_loss / len(train_loader)
        acc = epoch_rights / all_sample
        writer.add_scalar("avg_loss", avg_loss, epoch)
        writer.add_scalar("avg_acc", acc, epoch)

        if (epoch + 1) % 1 == 0:
            message = "Epoch{}, avg_loss={:.4f}, acc={:.4f}, time={}".format(
                epoch + 1,
                avg_loss, acc,
                time.ctime())
            print(message)
            if checkpoint_dir is not None:
                with open(log_file, 'a') as f:
                    f.write(message + '\n')


        if checkpoint_dir is not None and (epoch + 1) % 2 == 0:
            net.eval().cpu()
            ckpt_model_filename = "ckpt_epoch_" + str(epoch + 1) + ".pth"
            ckpt_model_path = os.path.join(checkpoint_dir, ckpt_model_filename)
            torch.save(net.state_dict(), ckpt_model_path)
            net.to(device).train()
    writer.close()

    # save model
    net.eval().cpu()
    save_model_filename = "final_epoch_" + str(epoch + 1) + ".model"
    save_model_path = os.path.join(checkpoint_dir, save_model_filename)
    torch.save(net.state_dict(), save_model_path)
    print("\nDone, trained model saved at", save_model_path)


def test(model_path):

    net = MyTDNN()
    net.load_state_dict(torch.load(model_path))
    test_db = Vox1_Test()

    # for enroll_data, test_data, label in test_db:
    #     enroll_data = enroll_data.unsqueeze(0).transpose(1, 2)
    #     # print(enroll_data.shape)
    #     test_data = test_data.unsqueeze(0).transpose(1, 2)
    #     enroll_o,enroll_a,enroll_b = net(enroll_data)
    #     test_o,test_a,test_b = net(test_data)
    #     # print(enroll_b.shape)
    #     # print(test_b.shape)
    #     cossim = torch.cosine_similarity(enroll_o, test_o)
    #     print(cossim)
    #
    # for thres in [0.01 * i + 0.8 for i in range(20)]:  # [0.8 ~ 1]
    #     cossim_thres = cossim > thres
    #     print(cossim_thres)

    def get_far(diff_same, same_same):
        far = diff_same / (diff_same + same_same)
        return far

    def get_frr(same_diff, diff_diff):
        if (same_diff+diff_diff) == 0:
            frr = 0
        else:
            frr = same_diff / (same_diff+diff_diff)
        return frr

    for thres in [0.001 * i + 0.9 for i in range(100)]:  # [0.9 ~ 1]
        same_same, same_diff, diff_same, diff_diff = 0, 0, 0, 0
        for i in range(100):
            # i = random.randint(1, len(test_db)-1)
            enroll_data, test_data, label = test_db[i]
            enroll_data = enroll_data.unsqueeze(0).transpose(1, 2)
            test_data = test_data.unsqueeze(0).transpose(1, 2)
            enroll_o, enroll_a, enroll_b = net(enroll_data)
            test_o, test_a, test_b = net(test_data)
            cos_o = torch.cosine_similarity(enroll_o, test_o)
            cos_a = torch.cosine_similarity(enroll_a, test_a)
            cos_b = torch.cosine_similarity(enroll_b, test_b)
            # print(cos_o, cos_a, cos_b)
            if label:
                if cos_a >= thres:
                    same_same += 1
                else:
                    same_diff += 1
            else:
                if cos_a >= thres:
                    diff_same += 1
                else:
                    diff_diff += 1
        far = get_far(diff_same, same_same)
        frr = get_frr(same_diff, diff_diff)
        print("Threshold={}:far={},frr={}".format(thres, far, frr))
def test2(model_path):

    net = MyTDNN()
    net.load_state_dict(torch.load(model_path))
    test_db = Vox1_Test()

    cosim_list = []

    for i in range(1000):
    # for enroll_data, test_data, label in test_db:
        i = random.randint(1, len(test_db)-1)
        enroll_data, test_data, label = test_db[i]
        enroll_data = enroll_data.unsqueeze(0).transpose(1, 2)
        test_data = test_data.unsqueeze(0).transpose(1, 2)
        enroll_o, enroll_a, enroll_b = net(enroll_data)
        test_o, test_a, test_b = net(test_data)
        cos_o = torch.cosine_similarity(enroll_o, test_o)
        cos_a = torch.cosine_similarity(enroll_a, test_a)
        cos_b = torch.cosine_similarity(enroll_b, test_b)

        cosim_list.append((cos_a, label))



    def get_far(diff_same, same_same):
        if (diff_same+same_same) == 0:
            far = 0
        else:
            far = diff_same / (diff_same + same_same)
        return far

    def get_frr(same_diff, diff_diff):
        if (same_diff+diff_diff) == 0:
            frr = 0
        else:
            frr = same_diff / (same_diff+diff_diff)
        return frr

    for thres in [0.001 * i + 0.9 for i in range(100)]:  # [0.9 ~ 1]
        same_same, same_diff, diff_same, diff_diff = 0, 0, 0, 0
        for cossim, label in cosim_list:
            if label:
                if cossim >= thres:
                    same_same += 1
                else:
                    same_diff += 1
            else:
                if cossim >= thres:
                    diff_same += 1
                else:
                    diff_diff += 1
        far = get_far(diff_same, same_same)
        frr = get_frr(same_diff, diff_diff)
        print("Threshold={}:far={},frr={}".format(thres, far, frr))


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("Training on ", device)
    # train(device)

    model_path = r'./checkpoints/final_epoch_10.model'
    test2(model_path)
    # acc = test(model_path)
    # print("final_acc=%f "% acc)
