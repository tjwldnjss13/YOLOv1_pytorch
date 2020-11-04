import torch
import torch.optim as optim
import torchvision.transforms as transforms

from model import YOLOv1, target_generator
from loss import yolo_custom_loss

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

    # Test sample
    import numpy as np
    import cv2 as cv
    import copy
    from PIL import Image

    # image = cv.imread('samples/pomeranian.jpg')
    # image = cv.resize(image, (224, 224))
    # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # image = torch.Tensor(image).unsqueeze(0).to(device)
    # image = image.permute((0, 3, 1, 2))

    image = Image.open('samples/pomeranian.jpg').convert('RGB')
    image_ = np.array(copy.deepcopy(image))
    image = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])(image)
    image = image.to(device)
    bbox = np.array([79, 191, 493, 602])
    if len(bbox.shape) == 1:
        bbox = np.array([bbox])
    label = np.array([1])

    bbox = torch.Tensor(bbox).to(device)
    label = torch.LongTensor(label).to(device)
    target = target_generator(bbox, label, 1, 2, (224, 224), (7, 7))

    import matplotlib.pyplot as plt
    for b in range(len(bbox)):
        image = cv.rectangle(image_, (bbox[b, 1], bbox[b, 0]), (bbox[b, 3], bbox[b, 2]), (0, 0, 255), thickness=5)
    plt.imshow(image_)
    # plt.show()

    # Real training with COCO dataset
    # root = ''
    # root_train = os.path.join(root, 'images', 'train')
    # root_val = os.path.join(root, 'images', 'val')
    # ann_train = os.path.join(root, 'annotations', 'instances_train2017.json')
    # ann_val = os.path.join(root, 'annotations', 'instances_val2017.json')
    #
    # dset_train = COCODataset(root_train, ann_train, transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))]))
    # dset_val = COCODataset(root_val, ann_val, transforms.Compose([transforms.ToTensor()]))
    #
    # train_data_loader = DataLoader(dset_train, batch_size, shuffle=True, collate_fn=collate_fn)
    # val_data_loader = DataLoader(dset_val, batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = optim.Adam(model.parameters(), lr)
    loss_func = yolo_custom_loss

    for e in range(num_epoch):
        output = model(image)
        loss = yolo_custom_loss(output, target, bbox.shape[0], 5, .5)
        print(loss)


        break