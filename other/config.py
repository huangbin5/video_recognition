import os
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
    def data_dir(database):
        if database == 'ucf101':
            raw_dir = '../dataraw/UCF101_'  # 原始数据的路径
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
