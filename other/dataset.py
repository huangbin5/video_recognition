import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from other.config import Path


# 继承Dataset的类需要重写 __len__ 和 __getitem__ 方法
class VideoDataset(Dataset):
    def __init__(self, dataset='ucf101', app='train', clip_len=16, preprocess=False):
        self.raw_dir, self.data_dir = Path.data_dir(dataset)
        app_dir = os.path.join(self.data_dir, app)
        self.app, self.clip_len = app, clip_len
        # 以下是C3D原文中的参数设置
        self.resize_height, self.resize_width, self.crop_size = 128, 171, 112

        if not os.path.exists(self.raw_dir):
            raise FileNotFoundError(f'Dataset {dataset} not found. You need to download it first.')
        # 提取视频帧
        if preprocess or (not self.__has_processed()):
            self.__preprocess()

        # 获取数据和类别
        self.videos, labels = [], []
        for action in sorted(os.listdir(app_dir)):
            for video in os.listdir(os.path.join(app_dir, action)):
                self.videos.append(os.path.join(app_dir, action, video))
                labels.append(action)
        print(f'Number of {app} videos: {len(self.videos):d}')
        # 将类别转化为用数字表示
        label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        self.label_ids = np.array([label2index[label] for label in labels], dtype=int)
        # 用于测试
        label_path = self.__get_path(f'../labels/{dataset}_labels.txt')
        if not os.path.exists(label_path):
            with open(label_path, 'w') as f:
                for i, label in enumerate(label2index):
                    f.writelines(str(i + 1) + ' ' + label + '\n')

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        # 数据预处理
        buffer = self.__load_frames(self.videos[index])
        buffer = self.__crop(buffer, self.clip_len, self.crop_size)
        # if self.app == 'train':
        #     # 数据增强，好像效果变差了。。。
        #     buffer = self.__randomflip(buffer)
        buffer = self.__normalize(buffer)
        buffer = self.__to_tensor(buffer)
        return torch.from_numpy(buffer), torch.tensor(self.label_ids[index])

    def __get_path(self, file):
        if file.startswith('./'):
            file = file[2:]
        return os.path.join(os.path.dirname(__file__), file)

    def __has_processed(self):
        train_dir = os.path.join(self.data_dir, 'train')
        if not os.path.exists(self.data_dir):
            return False
        elif not os.path.exists(train_dir):
            return False

        empty = True
        for i, video_class in enumerate(os.listdir(train_dir)):
            for video in os.listdir(os.path.join(self.data_dir, 'train', video_class)):
                empty = False
                video_dir = os.path.join(self.data_dir, 'train', video_class, video)
                frame_name = os.path.join(video_dir, sorted(os.listdir(video_dir))[0])
                image = cv2.imread(frame_name)
                if np.shape(image)[0] != self.resize_height or np.shape(image)[1] != self.resize_width:
                    return False
                break
            if i == 10:
                break
        return not empty

    def __preprocess(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            os.mkdir(os.path.join(self.data_dir, 'train'))
            os.mkdir(os.path.join(self.data_dir, 'val'))
            os.mkdir(os.path.join(self.data_dir, 'test'))

        for action in tqdm(os.listdir(self.raw_dir), desc='Preprocessing'):
            action_path = os.path.join(self.raw_dir, action)
            video_files = [video for video in os.listdir(action_path)]
            # 按6:2:2划分训练集、验证集和测试集
            train_and_valid, test_set = train_test_split(video_files, test_size=0.2, random_state=42)
            train_set, val_set = train_test_split(train_and_valid, test_size=0.2, random_state=42)

            train_dir, val_dir, test_dir = [
                os.path.join(self.data_dir, app, action) for app in ['train', 'val', 'test']]
            for app_set, app_dir in zip([train_set, val_set, test_set], [train_dir, val_dir, test_dir]):
                if not os.path.exists(app_dir):
                    os.mkdir(app_dir)
                for video in app_set:
                    self.__process_video(video, action, app_dir)

    def __process_video(self, video, action, action_dir):
        video_name = video.split('.')[0]
        video_dir = os.path.join(action_dir, video_name)
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        # 视频读取器
        capture = cv2.VideoCapture(os.path.join(self.raw_dir, action, video))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 最多每4帧读取1帧，如果帧数太少则减小读取频率，确保至少能够读到16帧
        EXTRACT_FREQUENCY = 4
        while EXTRACT_FREQUENCY > 1:
            if frame_count // EXTRACT_FREQUENCY > 16:
                break
            EXTRACT_FREQUENCY -= 1

        count, num = 0, 0
        while count < frame_count:
            _, frame = capture.read()
            # read有可能读取失败，可以通过设置读取位置，用retrieve方式读取
            if frame is None:
                capture.set(cv2.CAP_PROP_POS_FRAMES, num + 1)
                _, frame = capture.retrieve()
            assert frame is not None, f'Unknow error for {action}/{video}/{num + 1}'
            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(video_dir, '0000{}.jpg'.format(str(num))), img=frame)
                num += 1
            count += 1
        capture.release()

    def __load_frames(self, video_dir):
        frames = sorted([os.path.join(video_dir, img) for img in os.listdir(video_dir)])
        buffer = np.empty((len(frames), self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame in enumerate(frames):
            buffer[i] = np.array(cv2.imread(frame)).astype(np.float64)
        return buffer

    def __crop(self, buffer, clip_len, crop_size):
        # 帧不是完全随机选取，而是随机起始位置选取连续的帧
        t = np.random.randint(buffer.shape[0] - clip_len)
        # 空间起始位置也是随机的
        h = np.random.randint(buffer.shape[1] - crop_size)
        w = np.random.randint(buffer.shape[2] - crop_size)
        return buffer[t:t + clip_len, h:h + crop_size, w:w + crop_size, :]

    def __randomflip(self, buffer):
        if np.random.random() < 0.5:
            for i in range(buffer.shape[0]):
                buffer[i] = cv2.flip(buffer[i], flipCode=1)
        return buffer

    def __normalize(self, buffer):
        for i in range(buffer.shape[0]):
            # todo: 为什么用这3个数归一化？？？
            buffer[i] -= np.array([[[90.0, 98.0, 102.0]]])
        return buffer

    def __to_tensor(self, buffer):
        # (time, height, width, channel) -> (channel, time, height, width)
        return buffer.transpose((3, 0, 1, 2))


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_data = VideoDataset(dataset='ucf101', app='test', clip_len=8, preprocess=False)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)
        break
