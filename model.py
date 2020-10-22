import numpy as np
import torch
import torch.nn as nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class YOLOv1(nn.Module):
    def __init__(self, n_class, n_bbox):
        super(YOLOv1, self).__init__()
        self.n_class = n_class
        self.n_bbox = n_bbox
        self.train = True
        self.pretrain = False
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
        self.avgpool = nn.AvgPool2d(2, 2)
        self.leakyRelu = nn.LeakyReLU(.1)
        self.relu = nn.ReLU(True)
        self.softmax = nn.Softmax(dim=3)
        self.fc_pretrain = nn.Linear(7 * 7 * 1024, n_class)
        self.fc1 = nn.Linear(7 * 7 * 1024, 4096)
        self.fc2 = nn.Linear(4096, 7 * 7 * (5 * self.n_bbox + self.n_class))


    def forward(self, x):
        x = self.conv1_train(x) if self.train else self.conv1_detect(x)
        x = self.leakyRelu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.leakyRelu(x)
        x = self.maxpool(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.leakyRelu(x)
        x = self.maxpool(x)

        for _ in range(4):
            x = self.conv4_1(x)
            x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.leakyRelu(x)
        x = self.maxpool(x)

        for _ in range(2):
            x = self.conv5_1(x)
            x = self.conv5_2(x)

        if self.pretrain:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc_pretrain(x)
        else:
            x = self.conv5_3(x)
            x = self.conv5_4(x)
            x = self.leakyRelu(x)

            for _ in range(2):
                x = self.conv6(x)
            x = self.leakyRelu(x)

            x = x.contiguous().view(x.size(0), -1)
            x = self.fc1(x)
            x = self.leakyRelu(x)

            x = self.fc2(x)
            # x = self.relu(x)
            x = x.contiguous().view(x.size(0), 7, 7, -1)

            for i in range(self.n_bbox):
                x[:, :, :, 5 * i:5 * i + 4] = self.relu(x[:, :, :, 5 * i:5 * i + 4])
                x[:, :, :, 5 * i + 4] = self.leakyRelu(x[:, :, :, 5 * i + 4])
            x[:, :, :, 5 * self.n_bbox:] = self.softmax(x[:, :, :, 5 * self.n_bbox:])

        return x


def target_generator(bbox, class_, n_bbox, n_class, in_size, out_size):
    in_h, in_w = in_size[0], in_size[1]
    out_h, out_w = out_size[0], out_size[1]
    bbox_h, bbox_w = bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1]
    bbox_y, bbox_x = bbox[:, 0] + .5 * bbox_h, bbox[:, 1] + .5 * bbox_w

    objs = torch.zeros(out_size, dtype=torch.long).to(device)
    ratio = out_h / in_h
    bbox_y1_warp, bbox_x1_warp = torch.floor(bbox[:, 0] * ratio).int(), torch.floor(bbox[:, 1] * ratio).int()
    bbox_y2_warp, bbox_x2_warp = torch.ceil(bbox[:, 2] * ratio).int(), torch.ceil(bbox[:, 3] * ratio).int()
    for i in range(n_bbox):
        objs[bbox_y1_warp[i]:bbox_y2_warp[i] + 1, bbox_x1_warp[i]:bbox_x2_warp[i] + 1] = 1

    target = torch.zeros((out_h, out_w, 5 * n_bbox + n_class)).to(device)
    for i in range(n_bbox):
        target[:, :, 5 * i:5 * i + 4] = torch.Tensor([bbox_y[i], bbox_x[i], bbox_h[i], bbox_w[i]])
        target[:, :, 5 * i + 4] = objs
        for j in range(4):
            target[:, :, 5 * i + j] *= objs
    for y_ in range(out_h):
        for x_ in range(out_w):
            for c in class_:
                target[y_, x_, 5 * n_bbox + c * objs[y_, x_]] = 1

    return target




if __name__ == '__main__':
    import torch
    from torchsummary import summary
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = YOLOv1(20, 1).to(device)
    model.pretrain = False
    summary(model, (3, 224, 224))
    dummy = torch.zeros((1, 3, 224, 224)).to(device)
    out_ = model(dummy)
    print(out_.shape)
