import cv2
import os
import glob
import numpy as np

import torch
from torch import nn

from network.C3D_model import C3D
from utils.config import (DATASET, MODEL_NAME, get_device, Path)


def _get_checkpoint():
    CHECKPOINT_NOT_FOUND_ERROR = f'Checkpoint not found, you need to train {MODEL_NAME} first.'

    SAVE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log')
    runs = glob.glob(os.path.join(SAVE_ROOT, 'run_*'))
    run_ids = sorted([int(run.split('_')[-1]) for run in runs])
    assert run_ids, CHECKPOINT_NOT_FOUND_ERROR
    SAVE_DIR = os.path.join(SAVE_ROOT, 'run_' + str(run_ids[-1]))

    epochs = sorted(int(point.split('-')[-1][:-len('.pth')]) for point in os.listdir(SAVE_DIR)
                    if os.path.isfile(os.path.join(SAVE_DIR, point)))
    assert epochs, CHECKPOINT_NOT_FOUND_ERROR
    checkpoint = f'{MODEL_NAME}@{DATASET}_epoch-{epochs[-1]}.pth'
    print(f'Inference on checkpoint run_{run_ids[-1]}/{checkpoint}')
    return torch.load(os.path.join(SAVE_DIR, checkpoint), map_location=lambda storage, loc: storage)


def _preprocess(frame):
    frame = cv2.resize(frame, (171, 128))
    center_x, center_y = frame.shape[0] // 2, frame.shape[1] // 2
    frame = frame[center_x - 56:center_x + 56, center_y - 56:center_y + 56, :]
    return frame - np.array([[[90.0, 98.0, 102.0]]])


def _to_tensor(clip):
    # clip类型是list[ndarray]，实测直接用tensor初始化会很慢
    inputs = torch.tensor(np.array(clip, dtype=np.float32))
    inputs = inputs.permute((3, 0, 1, 2))
    return inputs.unsqueeze(dim=0)


def inference():
    DEVICE = get_device(is_test=True)
    with open(f'labels/{DATASET}_labels.txt', 'r') as f:
        index2label = f.readlines()
    assert MODEL_NAME == 'C3D', 'Can only test C3D model.'
    model = C3D(num_classes=101)
    model.to(DEVICE)
    model.load_state_dict(_get_checkpoint()['state_dict'])
    model.eval()

    # video = os.path.join(Path.data_dir(DATASET)[0], 'Basketball/v_Basketball_g01_c03.avi')
    video = os.path.join(os.path.dirname(__file__), 'data', '007.mp4')
    capture, clip = cv2.VideoCapture(video), []
    while True:
        has_more, frame = capture.read()
        if not has_more and frame is None:
            break
        clip.append(_preprocess(frame))
        # 预测每帧动作时使用的是之前的16帧
        if len(clip) == 16:
            inputs = _to_tensor(clip).to(DEVICE)
            with torch.no_grad():
                outputs = model(inputs)
            probs = nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().item()

            cv2.putText(frame, index2label[label].split(' ')[-1].strip(), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))
            cv2.putText(frame, "prob: %.4f" % probs[0][label], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))
            clip.pop(0)
        cv2.imshow('result', frame)
        cv2.waitKey(1)
    capture.release()


if __name__ == '__main__':
    inference()
