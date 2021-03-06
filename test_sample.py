import os
import cv2 as cv
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image

from model import YOLOv1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def detect_bbox(image, model):
    # Input
    # image: PIL image
    # model: pytorch model

    img = np.array(image)
    model = model.to(device)

    h_img, w_img, _ = image.shape()
    h_img_grid, w_img_grid = int(h_img / 7), int(w_img / 7)

    transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    img_tensor = transform(image)
    output = model(img_tensor)[0]

    for i in range(7):
        for j in range(7):
            for b in range(2):
                c = output[i, j, 5 * b + 4]

                if c > 0:
                    x = (i + output[i, j, 5 * b]) * w_img_grid
                    y = (j + output[i, j, 5 * b + 1]) * h_img_grid
                    w = output[i, j, 5 * b + 2] * w_img
                    h = output[i, j, 5 * b + 3] * h_img

                    x_min = int(x - .5 * w)
                    y_min = int(y - .5 * h)
                    x_max = int(x + .5 * w)
                    y_max = int(y + .5 * h)

                    cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), thickness=5)

    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    img_pth = 'samples/dogs.jpg'
    img = Image.open(img_pth)
    model = YOLOv1(21, 2)

    detect_bbox(img, model)


