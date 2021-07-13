import os


class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # origin data path
            root_dir = './dataraw/UCF101'
            # preprocess data path
            output_dir = './data/UCF101'
            return Path.get_path(root_dir), Path.get_path(output_dir)
        elif database == 'hmdb51':
            root_dir = '/Path/to/hmdb-51'
            output_dir = '/path/to/VAR/hmdb51'
            return Path.get_path(root_dir), Path.get_path(output_dir)
        else:
            print('Database {} not available.'.format(database))
            print('Currently only support ucf101 and hmdb51')
            raise NotImplementedError

    @staticmethod
    def model_dir():
        # model path
        return Path.get_path('./model/c3d-pretrained.pth')

    @staticmethod
    def get_path(file):
        if file.startswith('./'):
            file = file[2:]
        return os.path.join(os.path.dirname(__file__), file)
