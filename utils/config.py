import os
import argparse
import torch

# 已实现的数据集和模型
ALL_DATASETS = {'ucf101': 101, 'hmdb51': 51}
ALL_MODEL = ['C3D', 'R2Plus1D', 'R3D']

DATASET, MODEL_NAME = 'ucf101', 'C3D'
if DATASET not in ALL_DATASETS:
    raise NotImplementedError('Only ucf101 and hmdb51 datasets are supported.')
NUM_CLASSES = ALL_DATASETS[DATASET]
if MODEL_NAME not in ALL_MODEL:
    raise NotImplementedError('Only C3D, R2Plus1D and R3D models are supported.')


class Path(object):
    @staticmethod
    def data_dir(database, all_data):
        if database == 'ucf101':
            raw_dir = '../dataraw/UCF101'  # 原始数据的路径
            if not all_data:
                raw_dir += '_'
        elif database == 'hmdb51':
            raw_dir = '/Path/to/hmdb-51'
        else:
            raise NotImplementedError('Only ucf101 and hmdb51 datasets are supported.')
        data_dir = raw_dir.replace('raw', '')  # 处理后数据的路径
        return Path.__get_path(raw_dir), Path.__get_path(data_dir)

    @staticmethod
    def model_dir():
        return Path.__get_path('../model/c3d-pretrained.pth')  # 预训练模型参数

    @staticmethod
    def __get_path(file):
        if file.startswith('./'):
            file = file[2:]
        return os.path.join(os.path.dirname(__file__), file)


def get_device(is_test=False):
    if is_test:
        torch.backends.cudnn.benchmark = True
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device being used: {DEVICE}")
    return DEVICE


def get_hyper_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=15,
                        help='size of a batch (default: 15)')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=100,
                        help='epochs of training (default: 100)')
    parser.add_argument('--start_epoch', dest='start_epoch', type=int, default=0,
                        help='start epoch of training (default: 0)')
    parser.add_argument('--use_test', dest='use_test', type=bool, default=True,
                        help='use testset to evaluate (default: True)')
    parser.add_argument('--test_interval', dest='test_interval', type=int, default=1,
                        help='interval of USE_TEST (default: 1)')
    parser.add_argument('--ckpt_interval', dest='ckpt_interval', type=int, default=5,
                        help='interval of saving checkpoint (default: 5)')
    parser.add_argument('--all_data', dest='all_data', type=bool, default=False,
                        help='use all data for training (default: False)')
    parser.add_argument('--early_stop', dest='early_stop', type=int, default=10,
                        help='early stop steps without optimization (default: 10)')
    return parser.parse_args()
