import torch
import torch.optim as optim

from model import YOLOv1
from torchsummary import summary

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    lr = .0001
    batch_size = 64
    num_epoch = 10
    n_classes = 20

    model = YOLOv1(n_classes, True).to(device)

    import cv2 as cv

    image = cv.imread('samples/pomeranian.jpg')
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    bbox = [191, 79, 602, 493]

    # import matplotlib.pyplot as plt
    # image = cv.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), thickness=5)
    # plt.imshow(image)
    # plt.show()