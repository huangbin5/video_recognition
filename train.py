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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used: ", device)

dataset, model_name = 'ucf101', 'C3D'
if dataset not in ALL_DATASETS:
    raise NotImplementedError('Only ucf101 and hmdb51 datasets are supported.')
num_classes = ALL_DATASETS[dataset]

# start_epoch: 从已保存的哪个epoch接着训练
num_epochs, start_epoch, lr = 100, 0, 1e-3
# use_test: 训练过程中是否在测试集计算指标
use_test, test_interval, ckpt_interval = True, 5, 5

save_dir_root = os.path.dirname(os.path.abspath(__file__))
runs = glob.glob(os.path.join(save_dir_root, 'log', 'run_*'))
runs = sorted([int(run.split('_')[-1]) for run in runs])
run_id = runs[-1] if runs else 0
if start_epoch == 0:
    run_id += 1

save_dir = os.path.join(save_dir_root, 'log', 'run_' + str(run_id))
save_name = model_name + '-' + dataset


# 8G GPU support batch_size of at most 15
def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, batch_size=15, lr=lr,
                num_epochs=num_epochs, save_epoch=ckpt_interval, use_test=use_test, test_interval=test_interval):
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
    else:
        print('We only implemented C3D, R2Plus1D and R3D models.')
        raise NotImplementedError
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    model.to(device)
    criterion.to(device)

    if start_epoch == 0:
        print(f"Training {model_name} from scratch...")
    else:
        # Load checkpoint from "./log/run_0/C3D-ucf101_epoch-1.pth"
        ckpt_path = os.path.join(save_dir, save_name + '_epoch-' + str(start_epoch) + '.pth')
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        print(f"Initializing weights from: {ckpt_path}...")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print(f'Total params: {sum(p.numel() for p in model.parameters()) / 1000000.0:.2f}M')

    log_his = sorted(os.path.join(save_dir, t) for t in os.listdir(save_dir)
                     if os.path.isdir(os.path.join(save_dir, t)))
    last_log_dir = log_his[-1] if log_his else None
    log_dir = os.path.join(save_dir, datetime.now().strftime('%y%m%d_%H%M%S'))
    shutil.copytree(last_log_dir, log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    print(f'Training model on {dataset} dataset...')
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, app='train', clip_len=16),
                                  batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(VideoDataset(dataset=dataset, app='val', clip_len=16),
                                batch_size=batch_size, num_workers=4)
    test_dataloader = DataLoader(VideoDataset(dataset=dataset, app='test', clip_len=16),
                                 batch_size=batch_size, num_workers=4)

    train_val_loaders = {'train': train_dataloader, 'val': val_dataloader}
    train_val_sizes = {x: len(train_val_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    for epoch in range(start_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # Different modes primarily affect layers such as BatchNorm or Dropout.
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm(train_val_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                inputs = inputs.requires_grad_(True).to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                loss = criterion(outputs, labels)
                preds = torch.max(probs, 1)[1]

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / train_val_sizes[phase]
            epoch_acc = running_corrects.double() / train_val_sizes[phase]

            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()
                writer.add_scalar('data/train_loss', epoch_loss, epoch)
                writer.add_scalar('data/train_acc', epoch_acc, epoch)
            else:
                writer.add_scalar('data/val_loss', epoch_loss, epoch)
                writer.add_scalar('data/val_acc', epoch_acc, epoch)

            print(f"[{phase}] Epoch: {epoch + 1}/{num_epochs} Loss: {epoch_loss} Acc: {epoch_acc}")
            stop_time = timeit.default_timer()
            print(f"Execution time: {stop_time - start_time}\n")

        if (epoch + 1) % save_epoch == 0:
            save_path = os.path.join(save_dir, save_name + '_epoch-' + str(epoch + 1) + '.pth')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, save_path)
            print(f"Save model at {save_path}\n")

        if use_test and (epoch + 1) % test_interval == 0:
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('data/test_loss', epoch_loss, epoch)
            writer.add_scalar('data/test_acc', epoch_acc, epoch)

            print(f"[test] Epoch: {epoch + 1}/{num_epochs} Loss: {epoch_loss} Acc: {epoch_acc}")
            stop_time = timeit.default_timer()
            print(f"Execution time: {stop_time - start_time}\n")

    writer.close()


if __name__ == "__main__":
    train_model()
