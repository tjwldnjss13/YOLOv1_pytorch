import os
import sys
import torch
import torch.utils.data as data
import xml.etree.ElementTree as Et

from PIL import Image
from xml.etree.ElementTree import Element, ElementTree


class VOCDataset(data.Dataset):
    def __init__(self, root, is_validation=False, valid_split=.3, transforms=None, to_categorical=True):
        self.root = root
        self.is_validation = is_validation
        self.valid_split = valid_split
        self.transforms = transforms
        self.to_categorical = to_categorical
        self.class_dict = {'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8,
                           'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15,
                           'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}
        self.annotation = self.make_ann_list()

    def __getitem__(self, idx):
        img_dir = os.path.join(self.root, 'JPEGImages')
        ann = self.annotation[idx]
        img_fn = ann.find('filename').text
        img_path = os.path.join(img_dir, img_fn)
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = torch.as_tensor(img, dtype=torch.float64)

        obj = ann.findall('object')[0]
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        label = self.class_dict[name] if name in self.class_dict.keys() else 0
        bbox = [xmin, ymin, xmax, ymax]

        label = torch.as_tensor(label, dtype=torch.int64)
        bbox = torch.as_tensor(bbox, dtype=torch.float32)

        ann = {'label': label, 'bbox': bbox}

        return img, ann

    def __len__(self):
        return len(self.annotation)

    def make_ann_list(self):
        ann_dir = os.path.join(self.root, 'Annotations')
        anns_fn = os.listdir(ann_dir)

        anns_ = []
        for ann_fn in anns_fn:
            ann_path = os.path.join(ann_dir, ann_fn)
            ann = open(ann_path, 'r')
            tree = Et.parse(ann)
            root_ = tree.getroot()
            anns_.append(root_)
            ann.close()

        if self.is_validation:
            anns = anns_[int(len(anns_) * (1 - self.valid_split)):]
        else:
            anns = anns_[:int(len(anns_) * (1 - self.valid_split))]

        print('Annotations loaded!')

        return anns

    @staticmethod
    def to_categorical_pytorch(label, n_class):
        label_ = [0 for _ in range(n_class)]
        label_[label] = 1

        return label_


def collate_fn(batch):
    images_ = [item[0] for item in batch]
    anns_ = [item[1] for item in batch]

    return None


if __name__ == '__main__':
    root_dir = 'C://DeepLearningData/VOC2012/'
    train_dset = VOCDataset(root_dir, False)
    valid_dset = VOCDataset(root_dir, True)

    print(len(train_dset))
    print(len(valid_dset))
