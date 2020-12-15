is_debugging = False

import os
import time
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import YOLOv1
# from dataset.coco_dataset import COCODataset, collate_fn
from dataset.voc_dataset import VOCDataset, collate_fn
from utils import make_batch, make_annotation_batch, time_calculator
from loss import yolo_pretrain_custom_loss

if not torch.cuda.is_available():
    import os
    os.environ['KMP_DUPLICATE_LIB_OK' ] = 'TRUE'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    lr = .00005
    batch_size = 8
    num_epoch = 100
    n_class = 20
    n_bbox = 2
    model_save_term = 5

    model = YOLOv1(n_class, n_bbox).to(device)
    # PATH = 'yolov1_3e-05lr_100epoch_1.11530loss_0.75963acc.pth'
    # model = torch.load(PATH)
    model.train_ = True
    model.pretrain = True

    #################### COCO Dataset ####################
    # root = ''
    # root_train = os.path.join(root, 'images', 'train')
    # root_val = os.path.join(root, 'images', 'val')
    # ann_train = os.path.join(root, 'annotations', 'instances_train2017.json')
    # ann_val = os.path.join(root, 'annotations', 'instances_val20z17.json')
    #
    # dset_train = COCODataset(root_train, ann_train, transforms.Compose([transforms.ToTensor()]))
    # dset_val = COCODataset(root_val, ann_val, transforms.Compose([transforms.ToTensor()]))

    #################### VOC Dataset ####################
    root = 'C://DeepLearningData/VOC2012/'
    transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    dset_train = VOCDataset(root, is_validation=False, transforms=transforms, is_categorical=True)
    dset_val = VOCDataset(root, is_validation=True, transforms=transforms, is_categorical=True)

    train_data_loader = DataLoader(dset_train, batch_size, shuffle=True, collate_fn=collate_fn)
    val_data_loader = DataLoader(dset_val, batch_size, shuffle=True, collate_fn=collate_fn)

    n_train_data = len(dset_train)
    n_val_data = len(dset_val)

    optimizer = optim.SGD(model.parameters(), lr)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = yolo_pretrain_custom_loss

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    t_start = time.time()
    for e in range(num_epoch):
        n_images = 0
        n_batch = 0
        train_loss = 0
        train_acc = 0
        model.train()
        for i, (images, anns) in enumerate(train_data_loader):
            mini_batch = len(images)
            n_images += mini_batch
            n_batch += 1
            print('[{}/{}] {}/{} '.format(e + 1, num_epoch, n_images, n_train_data), end='')

            x_ = make_batch(images).to(device)
            y_ = make_annotation_batch(anns, 'label').to(device)

            output = model(x_)
            print(output.grad_fn)

            optimizer.zero_grad()
            loss = loss_func(output, y_)
            acc = (output.argmax(dim=1) == y_.argmax(dim=1)).sum()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += acc.item()

            print('<loss> {} <acc> {}'.format(loss.item(), acc.item() / mini_batch))

            if is_debugging:
                break

        if is_debugging:
            break

        train_losses.append(train_loss / n_batch)
        train_accs.append(train_acc / n_train_data)
        print('    <train_loss> {} <train_acc> {} '.format(train_losses[-1], train_accs[-1]), end='')

        val_loss = 0
        val_acc = 0
        n_images = 0
        model.eval()
        for i, (images, anns) in enumerate(val_data_loader):
            mini_batch = len(images)
            n_batch += 1
            x_ = make_batch(images).to(device)
            y_ = make_annotation_batch(anns, 'label').to(device)

            output = model(x_)

            loss = loss_func(output, y_)
            acc = (output.argmax(dim=1) == y_.argmax(dim=1)).sum()

            val_loss += loss.item()
            val_acc += acc.item()

        val_losses.append(val_loss / n_batch)
        val_accs.append(val_acc / n_val_data)

        print('<val_loss> {} <val_acc> {}'.format(val_losses[-1], val_accs[-1]))

        if (e + 1) % model_save_term == 0:
            PATH = 'pretrained models/yolov1_{}lr_{}epoch_{:.5f}loss_{:.5f}acc.pth'.format(lr, e + 1, val_losses[-1], val_accs[-1])
            torch.save(model, PATH)

    t_end = time.time()

    if not is_debugging:
        H, M, S = time_calculator(t_end - t_start)
        print('Train time : {}H {}M {:.2f}S'.format(H, M, S))

        plt.figure(1)
        plt.title('Train/Validation Loss')
        plt.plot([i for i in range(num_epoch)], train_losses, 'r-', label='train')
        plt.plot([i for i in range(num_epoch)], val_losses, 'b-', label='val')
        plt.legend()

        plt.figure(2)
        plt.title('Train/Validation Accuracy')
        plt.plot([i for i in range(num_epoch)], train_accs, 'r-', label='train')
        plt.plot([i for i in range(num_epoch)], val_accs, 'b-', label='val')
        plt.legend()
        plt.show()


