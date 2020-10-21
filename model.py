import torch.nn as nn


class YOLOv1(nn.Module):
    def __init__(self, n_classes, train=False):
        super(YOLOv1, self).__init__()
        self.n_classes = n_classes
        self.train = train
        self.conv1_train = nn.Conv2d(3, 64, 7, 1, 3)
        self.conv1_detect = nn.Conv2d(3, 64, 7, 2, 3)
        self.conv2 = nn.Conv2d(64, 192, 3, 1, 1)
        self.conv3_1 = nn.Conv2d(192, 128, 1, 1, 0)
        self.conv3_2 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 1, 1, 0)
        self.conv3_4 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_1 = nn.Conv2d(512, 256, 1, 1, 0)
        self.conv4_2 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 1, 1, 0)
        self.conv4_4 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.conv5_1 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.conv5_2 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.conv5_3 = self.conv6 = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.conv5_4 = nn.Conv2d(1024, 1024, 3, 2, 1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.activation = nn.LeakyReLU(.1)
        self.fc1 = nn.Linear(7 * 7 * 1024, 4096)
        self.fc2 = nn.Linear(4096, 7 * 7 * self.n_classes)

    def forward(self, x):
        x = self.conv1_train(x) if self.train else self.conv1_detect(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.activation(x)
        x = self.maxpool(x)

        for _ in range(4):
            x = self.conv4_1(x)
            x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.activation(x)
        x = self.maxpool(x)

        for _ in range(2):
            x = self.conv5_1(x)
            x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        x = self.activation(x)

        for _ in range(2):
            x = self.conv6(x)
        x = self.activation(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)
        x = x.view(x.size(0), 7, 7, -1)

        return x
