import numpy as np
import torch
import torch.nn as nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class YOLOv1(nn.Module):
    def __init__(self, n_class, n_bbox):
        super(YOLOv1, self).__init__()
        self.n_class = n_class
        self.n_bbox = n_bbox
        self.train_ = True
        self.pretrain = False
        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(.1, True)
        self.conv1_train = nn.Sequential(nn.Conv2d(3, 64, 7, 1, 3), nn.BatchNorm2d(64), self.lrelu)
        self.conv1_detect = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3), nn.BatchNorm2d(64), self.lrelu)
        self.conv2 = nn.Sequential(nn.Conv2d(64, 192, 3, 1, 1), nn.BatchNorm2d(192), self.lrelu)
        self.conv3_1 = nn.Sequential(nn.Conv2d(192, 128, 1, 1, 0), nn.BatchNorm2d(128), self.lrelu)
        self.conv3_2 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), self.lrelu)
        self.conv3_3 = nn.Sequential(nn.Conv2d(256, 256, 1, 1, 0), nn.BatchNorm2d(256), self.lrelu)
        self.conv3_4 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), self.lrelu)
        self.conv4_1 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0), nn.BatchNorm2d(256), self.lrelu)
        self.conv4_2 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), self.lrelu)
        self.conv4_3 = nn.Sequential(nn.Conv2d(512, 512, 1, 1, 0), nn.BatchNorm2d(512), self.lrelu)
        self.conv4_4 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1), nn.BatchNorm2d(1024), self.lrelu)
        self.conv5_1 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0), nn.BatchNorm2d(512), self.lrelu)
        self.conv5_2 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1), nn.BatchNorm2d(1024), self.lrelu)
        self.conv5_3 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1), nn.BatchNorm2d(1024), self.lrelu)
        self.conv5_4 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 2, 1), nn.BatchNorm2d(1024), self.lrelu)
        self.conv6 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1), nn.BatchNorm2d(1024), self.lrelu)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(7, 7)
        self.softmax_pretrain = nn.Softmax(dim=1)
        self.softmax = nn.Softmax(dim=3)
        self.fc1_pretrain = nn.Linear(4096, 512)
        self.fc2_pretrain = nn.Linear(512, n_class)
        self.fc1 = nn.Linear(7 * 7 * 1024, 4096)
        self.fc2 = nn.Linear(4096, 7 * 7 * (5 * self.n_bbox + self.n_class))

    def forward(self, x):
        x = self.lrelu(x)
        x = self.relu(x)
        x = self.conv1_train(x) if self.train else self.conv1_detect(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.maxpool(x)

        for _ in range(4):
            x = self.conv4_1(x)
            x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.maxpool(x)

        for _ in range(2):
            x = self.conv5_1(x)
            x = self.conv5_2(x)

        if self.pretrain:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc1_pretrain(x)
            x = self.fc2_pretrain(x)
            x = self.softmax_pretrain(x)
        else:
            x = self.conv5_3(x)
            x = self.conv5_4(x)

            for _ in range(2):
                x = self.conv6(x)

            x = x.contiguous().view(x.size(0), -1)
            x = self.fc1(x)
            x = self.lrelu(x)

            x = self.fc2(x)
            # x = self.relu(x)
            x = x.contiguous().view(x.size(0), 7, 7, -1)

            x = self.lrelu(x)
            # bbox 좌표 부분은 0~1로 normalize 해야 함


            # for i in range(self.n_bbox):
            #     x[:, :, :, 5 * i:5 * i + 4] = self.relu(x[:, :, :, 5 * i:5 * i + 4])
            #     x[:, :, :, 5 * i + 4] = self.lrelu(x[:, :, :, 5 * i + 4])
            # x[:, :, :, 5 * self.n_bbox:] = self.softmax(x[:, :, :, 5 * self.n_bbox:])

        return x

