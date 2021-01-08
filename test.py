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

    class_dict = {'background': 0, 'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 'bus': 6, 'car': 7,
                  'cat': 8, 'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15,
                  'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20, 'ambigious': 255}
    key_list = list(class_dict.keys())
    value_list = list(class_dict.values())

    img = np.array(image)
    model = model.to(device)

    h_img, w_img, _ = img.shape
    h_img_grid, w_img_grid = int(h_img / 7), int(w_img / 7)

    transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(device)
    output = model(img_tensor)[0]

    for i in range(7):
        for j in range(7):
            for b in range(2):
                c = output[i, j, 5 * b + 4]

                if c > .995:
                    x = (i + output[i, j, 5 * b]) * w_img_grid
                    y = (j + output[i, j, 5 * b + 1]) * h_img_grid
                    # w = output[i, j, 5 * b + 2]
                    # h = output[i, j, 5 * b + 3]
                    w = output[i, j, 5 * b + 2] * w_img
                    h = output[i, j, 5 * b + 3] * h_img
                    # print(c, x.item(), y.item(), w.item(), h.item())

                    x_min = int(x - .5 * w)
                    y_min = int(y - .5 * h)
                    x_max = int(x + .5 * w)
                    y_max = int(y + .5 * h)

                    cls = torch.argmax(output[i, j, 5 * b:])
                    cls_idx = value_list.index(cls)
                    cls_name = key_list[cls_idx]

                    text = cls_name + ':{:.2f}'.format(c)

                    cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), thickness=5)
                    cv.putText(img, text, (x_min, y_min), 2, 1, (0, 0, 0), 3)
                    break

    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    img_pth = 'samples/dogs.jpg'
    model_pth = 'trained models/base model/???.pth'

    img = Image.open(img_pth)
    model = torch.load(model_pth)
    model.train_ = False

    detect_bbox(img, model)


