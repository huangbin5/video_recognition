import time
from datetime import datetime
import os
import glob
import shutil
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils.dataset import VideoDataset
from utils.config import (DATASET, MODEL_NAME, NUM_CLASSES, get_device, get_hyper_parameter)
from network.C3D_model import C3D
from network import R2Plus1D_model, R3D_model

'''
batch_size: 8G GPU支持 batch_size 最大为 15
lr: 学习率
num_epochs: 迭代总epoch数
start_epoch: 从已保存的哪个epoch接着训练
use_test: 训练过程中是否在测试集计算指标
test_interval: 使用测试集计算的间隔
ckpt_interval: 保存模型参数的间隔
all_data: 是否使用全部数据训练。在调代码时使用部分数据集更加方便
'''
Hyper = get_hyper_parameter()
# 尝试使用GPU
DEVICE = get_device()
# 获取本次run保存的路径
SAVE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log')
runs = glob.glob(os.path.join(SAVE_ROOT, 'run_*'))
run_ids = sorted([int(run.split('_')[-1]) for run in runs])
run_id = run_ids[-1] if run_ids else 0
if Hyper.start_epoch == 0:
    run_id += 1
SAVE_DIR = os.path.join(SAVE_ROOT, 'run_' + str(run_id))
MIN_LOSS, EARLY_STOP = None, 0


def _init_model(lr=Hyper.lr):
    if MODEL_NAME == 'C3D':
        model = C3D(num_classes=NUM_CLASSES, pretrained=True)
        train_params = [{'params': model.get_1x_lr_params(), 'lr': lr},
                        {'params': model.get_10x_lr_params(), 'lr': lr * 10}]
    elif MODEL_NAME == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=NUM_CLASSES, layer_sizes=(2, 2, 2, 2))
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif MODEL_NAME == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=NUM_CLASSES, layer_sizes=(2, 2, 2, 2))
        train_params = model.parameters()
    return model, train_params


def _load_param_log(ckpt_path, log_dir, model, optimizer):
    # 加载参数
    print(f"Initializing weights from: {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['opt_dict'])
    # 合并日志
    log_his = sorted(os.path.join(SAVE_DIR, log) for log in os.listdir(SAVE_DIR)
                     if os.path.isdir(os.path.join(SAVE_DIR, log)))
    last_log_dir = log_his[-1] if log_his else None
    shutil.copytree(last_log_dir, log_dir)


def _run_epoch(phase, epoch, data_loader, writer, model, criterion, optimizer=None, scheduler=None):
    time.sleep(0.05)
    # 设置训练或测试模式
    model.train() if phase == 'train' else model.eval()
    total_loss, total_acc = 0.0, 0.0

    # 按batch迭代
    for inputs, labels in tqdm(data_loader, desc=f'Epoch {epoch}/{Hyper.num_epochs} [{phase}]'):
        inputs = inputs.requires_grad_('train' == phase).to(DEVICE)
        labels = labels.to(DEVICE)
        if phase == 'train':
            optimizer.zero_grad()
            outputs = model(inputs)
        else:
            with torch.no_grad():
                outputs = model(inputs)
        loss = criterion(outputs, labels)
        if phase == 'train':
            loss.backward()
            optimizer.step()

        # 计算指标
        total_loss += loss.item() * inputs.size(0)
        preds = torch.max(nn.Softmax(dim=1)(outputs), dim=1)[1]
        total_acc += torch.sum(preds == labels.data)
    epoch_loss = total_loss / len(data_loader.dataset)
    epoch_acc = total_acc.double() / len(data_loader.dataset)
    # 记录每个epoch的指标
    writer.add_scalar(f'data/{phase}_loss', epoch_loss, epoch)
    writer.add_scalar(f'data/{phase}_acc', epoch_acc, epoch)
    print(f"Loss: {epoch_loss}  Acc: {epoch_acc}\n")

    if phase == 'train':
        # 衰减学习率
        scheduler.step()
    elif phase == 'val':
        global MIN_LOSS, EARLY_STOP
        if MIN_LOSS is None or epoch_loss < MIN_LOSS:
            # 保存验证集上损失更小的模型
            MIN_LOSS, EARLY_STOP = epoch_loss, 0
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict()
            }, os.path.join(SAVE_DIR, f'{MODEL_NAME}@{DATASET}_best.pth.tar'))
            print('Save best model.')
        else:
            EARLY_STOP += 1


def train(start_epoch=Hyper.start_epoch, batch_size=Hyper.batch_size):
    # 加载数据
    train_dataloader = DataLoader(VideoDataset(dataset=DATASET, app='train', clip_len=16, all_data=Hyper.all_data),
                                  batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(VideoDataset(dataset=DATASET, app='val', clip_len=16, all_data=Hyper.all_data),
                                batch_size=batch_size, num_workers=4)
    test_dataloader = DataLoader(VideoDataset(dataset=DATASET, app='test', clip_len=16, all_data=Hyper.all_data),
                                 batch_size=batch_size, num_workers=4)
    train_val_loaders = {'train': train_dataloader, 'val': val_dataloader}

    # 模型、策略、算法
    model, train_params = _init_model()
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    criterion.to(DEVICE)
    optimizer = optim.SGD(train_params, lr=Hyper.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 恢复模型和log
    model_data_name = f'{MODEL_NAME}@{DATASET}'
    log_dir = os.path.join(SAVE_DIR, datetime.now().strftime('%y%m%d_%H%M%S'))
    if start_epoch == 0:
        print(f"Training {model_data_name} from scratch...")
    else:
        ckpt_path = os.path.join(SAVE_DIR, f'{model_data_name}_epoch-{str(start_epoch)}.pth.tar')
        _load_param_log(ckpt_path, log_dir, model, optimizer)
    print(f'Total params: {sum(p.numel() for p in model.parameters()) / 1000000.0:.2f}M')

    with SummaryWriter(log_dir=log_dir) as writer:
        for epoch in range(start_epoch, Hyper.num_epochs):
            epoch += 1
            # 训练并验证epoch
            for phase in ['train', 'val']:
                _run_epoch(phase, epoch, train_val_loaders[phase], writer, model, criterion, optimizer, scheduler)
                # 模型已经不能再优化了，提前结束训练
                if EARLY_STOP >= Hyper.early_stop:
                    print('There is no optimization for a long time, stop training!!!')
                    return
            # 测试模型
            if Hyper.use_test and epoch % Hyper.test_interval == 0:
                _run_epoch('test', epoch, test_dataloader, writer, model, criterion)
            # 保存模型
            if epoch % Hyper.ckpt_interval == 0:
                save_path = os.path.join(SAVE_DIR, f'{model_data_name}_epoch-{str(epoch)}.pth.tar')
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'opt_dict': optimizer.state_dict()
                }, save_path)
                print(f"Save model at {save_path}\n")


if __name__ == "__main__":
    train()
