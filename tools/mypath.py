import os


class Path(object):
    @staticmethod
    def data_dir(database):
        if database == 'ucf101':
            raw_dir = '../dataraw/UCF101'  # 原始数据的路径
            root_dir = '../data/UCF101'  # 处理后数据的路径
            return Path.__get_path(raw_dir), Path.__get_path(root_dir)
        elif database == 'hmdb51':
            raw_dir = '/Path/to/hmdb-51'
            root_dir = '/path/to/VAR/hmdb51'
            return Path.__get_path(raw_dir), Path.__get_path(root_dir)
        else:
            raise NotImplementedError('Only ucf101 and hmdb51 datasets are supported.')

    @staticmethod
    def model_dir():
        return Path.__get_path('../model/c3d-pretrained.pth')  # 预训练模型参数

    @staticmethod
    def __get_path(file):
        if file.startswith('./'):
            file = file[2:]
        return os.path.join(os.path.dirname(__file__), file)
