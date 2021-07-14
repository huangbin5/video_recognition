import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from tools.mypath import Path


# classes extended Dataset must rewrite __len__ and __getitem__ methods
class VideoDataset(Dataset):
    def __init__(self, dataset='ucf101', app='train', clip_len=16, preprocess=False):
        self.raw_dir, self.root_dir = Path.data_dir(dataset)
        app_dir = os.path.join(self.root_dir, app)
        self.clip_len = clip_len
        self.app = app

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        if not os.path.exists(self.raw_dir):
            raise RuntimeError('Dataset not found or corrupted. You need to download it from official website.')

        if (not self.__has_preprocess()) or preprocess:
            print(f'Preprocessing of {dataset} dataset, this will take long, but it will be done only once.')
            self.__preprocess()

        # Get all the videos and its label
        self.videos, labels = [], []
        for label in sorted(os.listdir(app_dir)):
            for fname in os.listdir(os.path.join(app_dir, label)):
                self.videos.append(os.path.join(app_dir, label, fname))
                labels.append(label)
        assert len(labels) == len(self.videos)
        print(f'Number of {app} videos: {len(self.videos):d}')

        # Match each label with an index
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label into its index
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if dataset == "ucf101":
            if not os.path.exists('../labels/ucf_labels.txt'):
                with open('../labels/ucf_labels.txt', 'w') as f:
                    for ii, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(ii + 1) + ' ' + label + '\n')
        elif dataset == 'hmdb51':
            if not os.path.exists('../labels/hmdb_labels.txt'):
                with open('../labels/hmdb_labels.txt', 'w') as f:
                    for ii, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(ii + 1) + ' ' + label + '\n')

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.videos[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        if self.app == 'test':
            # Perform data augmentation
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def __has_preprocess(self):
        # TODO: Check image size in output_dir
        train_dir = os.path.join(self.root_dir, 'train')
        if not os.path.exists(self.root_dir):
            return False
        elif not os.path.exists(train_dir):
            return False

        empty = True
        for ii, video_class in enumerate(os.listdir(train_dir)):
            for video in os.listdir(os.path.join(self.root_dir, 'train', video_class)):
                empty = False
                video_dir = os.path.join(self.root_dir, 'train', video_class, video)
                frame_name = os.path.join(video_dir, sorted(os.listdir(video_dir))[0])
                image = cv2.imread(frame_name)
                if np.shape(image)[0] != 128 or np.shape(image)[1] != 171:
                    return False
                break
            if ii == 10:
                break
        return not empty

    def __preprocess(self):
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
            os.mkdir(os.path.join(self.root_dir, 'train'))
            os.mkdir(os.path.join(self.root_dir, 'val'))
            os.mkdir(os.path.join(self.root_dir, 'test'))

        # Split train/val/test sets
        for action in os.listdir(self.raw_dir):
            action_path = os.path.join(self.raw_dir, action)
            video_files = [video for video in os.listdir(action_path)]

            train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
            train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

            train_dir, val_dir, test_dir = [
                os.path.join(self.root_dir, app, action) for app in ['train', 'val', 'test']]
            for app, app_dir in zip([train, val, test], [train_dir, val_dir, test_dir]):
                if not os.path.exists(app_dir):
                    os.mkdir(app_dir)
                for video in app:
                    self.__process_video(video, action, app_dir)
        print('Preprocessing finished.')

    def __process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        video_dir = os.path.join(save_dir, video_filename)
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)

        capture = cv2.VideoCapture(os.path.join(self.raw_dir, action_name, video))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 4
        while EXTRACT_FREQUENCY > 1:
            if frame_count // EXTRACT_FREQUENCY > 16:
                break
            EXTRACT_FREQUENCY -= 1

        count, num = 0, 0
        while count < frame_count:
            _, frame = capture.read()
            if frame is None:
                break
            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(num))), img=frame)
                num += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[
                 time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]
        return buffer


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_data = VideoDataset(dataset='ucf101', app='test', clip_len=8, preprocess=False)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break
