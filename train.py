import numpy as np
import torch
import torch.optim as optim

from model import YOLOv1, target_generator
from loss import yolo_custom_loss
from torchsummary import summary

if not torch.cuda.is_available():
    import os
    os.environ['KMP_DUPLICATE_LIB_OK' ] = 'TRUE'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    lr = .0001
    batch_size = 64
    num_epoch = 10
    n_class = 2
    n_bbox = 1

    model = YOLOv1(n_class, n_bbox).to(device)

    import cv2 as cv

    image = cv.imread('samples/pomeranian.jpg')
    image = cv.resize(image, (224, 224))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    bbox = np.array([79, 191, 493, 602])
    if len(bbox.shape) == 1:
        bbox = np.array([bbox])
    label = np.array([1])

    image = torch.Tensor(image).unsqueeze(0).to(device)
    image = image.permute((0, 3, 1, 2))
    bbox = torch.Tensor(bbox).to(device)
    label = torch.LongTensor(label).to(device)

    target = target_generator(bbox, label, 1, 2, (224, 224), (7, 7))


    # import matplotlib.pyplot as plt
    # image = cv.rectangle(image, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 0, 255), thickness=5)
    # plt.imshow(image)
    # plt.show()

    for e in range(num_epoch):
        output = model(image)
        loss = yolo_custom_loss(output, target, bbox.shape[0], 5, .5)
        print(loss)

        break