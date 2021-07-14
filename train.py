import timeit
from datetime import datetime
import os
import glob
import shutil
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from tools.dataset import VideoDataset
from network import C3D_model, R2Plus1D_model, R3D_model

# 已实现的数据集和模型
ALL_DATASETS = {'ucf101': 101, 'hmdb51': 51}
ALL_MODEL = ['C3D', 'R2Plus1D', 'R3D']

# 尝试使用GPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device being used: {DEVICE}")

DATASET, MODEL_NAME = 'ucf101', 'C3D'
if DATASET not in ALL_DATASETS:
    raise NotImplementedError('Only ucf101 and hmdb51 datasets are supported.')
NUM_CLASSES = ALL_DATASETS[DATASET]
if MODEL_NAME not in ALL_MODEL:
    raise NotImplementedError('Only C3D, R2Plus1D and R3D models are supported.')

# start_epoch: 从已保存的哪个epoch接着训练
num_epochs, start_epoch, lr = 100, 0, 1e-3
# use_test: 训练过程中是否在测试集计算指标
use_test, test_interval, ckpt_interval = True, 5, 5

# 获取本次run保存的路径
SAVE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log')
runs = glob.glob(os.path.join(SAVE_ROOT, 'run_*'))
runs = sorted([int(run.split('_')[-1]) for run in runs])
run_id = runs[-1] if runs else 0
if start_epoch == 0:
    run_id += 1
SAVE_DIR = os.path.join(SAVE_ROOT, 'run_' + str(run_id))


def init_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, lr=lr):
    if model_name == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=True)
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif model_name == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif model_name == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = model.parameters()
    return model, train_params


def load_model_log(save_dir, ckpt_path, log_dir, model, optimizer):
    print(f"Initializing weights from: {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['opt_dict'])

    log_his = sorted(os.path.join(save_dir, t) for t in os.listdir(save_dir)
                     if os.path.isdir(os.path.join(save_dir, t)))
    last_log_dir = log_his[-1] if log_his else None
    shutil.copytree(last_log_dir, log_dir)


def train_epoch(epoch, train_val_loaders, model, criterion, optimizer, scheduler, writer):
    for phase in ['train', 'val']:
        train_val_sizes = len(train_val_loaders[phase].dataset)
        # 设置训练或测试模式
        model.train() if phase == 'train' else model.eval()
        start_time = timeit.default_timer()
        total_loss, total_acc = 0.0, 0.0

        # 按batch迭代
        for inputs, labels in tqdm(train_val_loaders[phase]):
            inputs = inputs.requires_grad_(True).to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            if phase == 'train':
                outputs = model(inputs)
            else:
                with torch.no_grad():
                    outputs = model(inputs)
            loss = criterion(outputs, labels)
            if phase == 'train':
                loss.backward()
                optimizer.step()
            # 计算指标
            preds = torch.max(nn.Softmax(dim=1)(outputs), dim=1)[1]
            total_loss += loss.item() * inputs.size(0)
            total_acc += torch.sum(preds == labels.data)
        epoch_loss = total_loss / train_val_sizes
        epoch_acc = total_acc.double() / train_val_sizes

        # 记录每个epoch的指标
        if phase == 'train':
            # 衰减学习率
            scheduler.step()
            writer.add_scalar('data/train_loss', epoch_loss, epoch)
            writer.add_scalar('data/train_acc', epoch_acc, epoch)
        else:
            writer.add_scalar('data/val_loss', epoch_loss, epoch)
            writer.add_scalar('data/val_acc', epoch_acc, epoch)

        print(f"[{phase}] Epoch: {epoch + 1}/{num_epochs} Loss: {epoch_loss} Acc: {epoch_acc}")
        stop_time = timeit.default_timer()
        print(f"Execution time: {stop_time - start_time}\n")


# 8G GPU支持 batch_size 最大为 15
def train_model(dataset=DATASET, model_name=MODEL_NAME, num_classes=NUM_CLASSES, save_dir=SAVE_DIR,
                batch_size=15, num_epochs=num_epochs, start_epoch=start_epoch, lr=lr,
                use_test=use_test, test_interval=test_interval, ckpt_interval=ckpt_interval):
    # 模型、策略、算法
    model, train_params = init_model()
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    criterion.to(DEVICE)
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 恢复模型和log
    model_data_name = model_name + '@' + dataset
    log_dir = os.path.join(save_dir, datetime.now().strftime('%y%m%d_%H%M%S'))
    if start_epoch == 0:
        print(f"Training {model_data_name} from scratch...")
    else:
        ckpt_path = os.path.join(save_dir, model_data_name + '_epoch-' + str(start_epoch) + '.pth')
        load_model_log(save_dir, ckpt_path, log_dir, model, optimizer)
    print(f'Total params: {sum(t.numel() for t in model.parameters()) / 1000000.0:.2f}M')

    # 加载数据
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, app='train', clip_len=16),
                                  batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(VideoDataset(dataset=dataset, app='val', clip_len=16),
                                batch_size=batch_size, num_workers=4)
    test_dataloader = DataLoader(VideoDataset(dataset=dataset, app='test', clip_len=16),
                                 batch_size=batch_size, num_workers=4)
    train_val_loaders = {'train': train_dataloader, 'val': val_dataloader}
    test_size = len(test_dataloader.dataset)

    writer = SummaryWriter(log_dir=log_dir)
    for epoch in range(start_epoch, num_epochs):
        train_epoch(epoch, train_val_loaders, model, criterion, optimizer, scheduler, writer)
        if (epoch + 1) % ckpt_interval == 0:
            save_path = os.path.join(save_dir, model_data_name + '_epoch-' + str(epoch + 1) + '.pth')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict()
            }, save_path)
            print(f"Save model at {save_path}\n")

        if use_test and (epoch + 1) % test_interval == 0:
            model.eval()
            start_time = timeit.default_timer()

            total_loss = 0.0
            total_acc = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                total_acc += torch.sum(preds == labels.data)

            epoch_loss = total_loss / test_size
            epoch_acc = total_acc.double() / test_size

            writer.add_scalar('data/test_loss', epoch_loss, epoch)
            writer.add_scalar('data/test_acc', epoch_acc, epoch)

            print(f"[test] Epoch: {epoch + 1}/{num_epochs} Loss: {epoch_loss} Acc: {epoch_acc}")
            stop_time = timeit.default_timer()
            print(f"Execution time: {stop_time - start_time}\n")

    writer.close()


if __name__ == "__main__":
    train_model()
