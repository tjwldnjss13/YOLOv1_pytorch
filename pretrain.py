import os
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import YOLOv1
# from dataset.coco_dataset import COCODataset, collate_fn
from dataset.voc_dataset import VOCDataset, collate_fn
from loss import yolo_custom_loss
from utils import make_batch, make_annotation_batch

if not torch.cuda.is_available():
    import os
    os.environ['KMP_DUPLICATE_LIB_OK' ] = 'TRUE'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    lr = .0001
    batch_size = 128
    num_epoch = 10
    n_class = 21
    n_bbox = 1

    model = YOLOv1(n_class, n_bbox).to(device)
    model.train = True
    model.pretrain = True

    #################### COCO Dataset ####################
    # root = ''
    # root_train = os.path.join(root, 'images', 'train')
    # root_val = os.path.join(root, 'images', 'val')
    # ann_train = os.path.join(root, 'annotations', 'instances_train2017.json')
    # ann_val = os.path.join(root, 'annotations', 'instances_val2017.json')
    #
    # dset_train = COCODataset(root_train, ann_train, transforms.Compose([transforms.ToTensor()]))
    # dset_val = COCODataset(root_val, ann_val, transforms.Compose([transforms.ToTensor()]))

    #################### VOC Dataset ####################
    root = 'C://DeepLearningData/VOC2012/'
    transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    dset_train = VOCDataset(root, False, transforms=transforms, to_categorical=True)
    dset_val = VOCDataset(root, True, transforms=transforms, to_categorical=True)

    train_data_loader = DataLoader(dset_train, batch_size, shuffle=True, collate_fn=collate_fn)
    val_data_loader = DataLoader(dset_val, batch_size, shuffle=True, collate_fn=collate_fn)

    num_objs = len(dset_train)

    optimizer = optim.SGD(model.parameters(), lr)
    loss_func =

    for e in range(num_epoch):
        n_images = 0
        for i, (images, anns) in enumerate(train_data_loader):
            mini_batch = len(images)
            n_images += mini_batch
            print('[{}/{}] {}/{}'.format(e + 1, num_epoch, n_images, num_objs), end='')

            x_ = make_batch(images).to(device)
            y_ = make_annotation_batch(anns, 'bbox').to(device)

            output = model(x_)

            optimizer.zero_grad()
            loss = loss_func(output, y_)


