import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloaders.dataset import VideoDataset
from network import C3D_model, R2Plus1D_model, R3D_model

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used: ", device)

num_epochs = 100  # Number of epochs for training
resume_epoch = 15  # From which epoch to continue training
use_test = True  # See evolution of the test set when training
test_interval = 10  # Run on test set every test_interval epochs
snapshot = 1  # Store a model every snapshot epochs
lr = 1e-3  # Learning rate

dataset = 'ucf101'  # Options: ucf101 or hmdb51
if dataset == 'hmdb51':
    num_classes = 51
elif dataset == 'ucf101':
    num_classes = 101
else:
    print('We only implemented ucf101 and hmdb51 datasets.')
    raise NotImplementedError

save_dir_root = os.path.dirname(os.path.abspath(__file__))

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    # todo Not understand
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
model_name = 'C3D'  # Options: C3D or R2Plus1D or R3D
save_name = model_name + '-' + dataset


# 8G GPU support batch_size of at most 15
def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, batch_size=15, lr=lr,
                num_epochs=num_epochs, save_epoch=snapshot, use_test=use_test, test_interval=test_interval):
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

    if resume_epoch == 0:
        print(f"Training {model_name} from scratch...")
    else:
        # Load checkpoint from "./run/run_0/models/C3D-ucf101_epoch-1.pth.tar"
        path = os.path.join(save_dir, 'models', save_name + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        print(f"Initializing weights from: {path}...")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print(f'Total params: {sum(p.numel() for p in model.parameters()) / 1000000.0:.2f}M')

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
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

    for epoch in range(resume_epoch, num_epochs):
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
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)

            print(f"[{phase}] Epoch: {epoch + 1}/{num_epochs} Loss: {epoch_loss} Acc: {epoch_acc}")
            stop_time = timeit.default_timer()
            print(f"Execution time: {stop_time - start_time}\n")

        if (epoch + 1) % save_epoch == 0:
            save_path = os.path.join(save_dir, 'models', save_name + '_epoch-' + str(epoch) + '.pth.tar')
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

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

            print(f"[test] Epoch: {epoch + 1}/{num_epochs} Loss: {epoch_loss} Acc: {epoch_acc}")
            stop_time = timeit.default_timer()
            print(f"Execution time: {stop_time - start_time}\n")

    writer.close()


if __name__ == "__main__":
    train_model()
