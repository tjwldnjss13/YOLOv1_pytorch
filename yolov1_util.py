import torch
import numpy as np


def generate_target_batch(annotation, n_bbox_predict, n_class, in_size, out_size):
    n_ann = len(annotation)

    target_tensors = []
    for i in range(n_ann):
        ann = annotation[i]
        classes, bboxes = ann['class'], ann['bbox']

        target_tensor = make_target_tensor(bboxes, classes, n_bbox_predict, n_class, in_size, out_size)
        target_tensors.append(target_tensor)

    target_batch = make_batch(target_tensors)

    return target_batch


def make_target_tensor(bboxes, classes, n_bbox_predict, n_class, in_size, out_size):
    n_gt = len(bboxes)
    in_h, in_w = in_size[0], in_size[1]
    out_h, out_w = out_size[0], out_size[1]
    # hs, ws = bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]
    # ys, xs = bboxes[:, 0] + .5 * hs, bboxes[:, 1] + .5 * ws

    ratio = out_h / in_h
    # bbox_y1_warp, bbox_x1_warp = torch.floor(bbox[:, 0] * ratio).int(), torch.floor(bbox[:, 1] * ratio).int()
    # bbox_y2_warp, bbox_x2_warp = torch.ceil(bbox[:, 2] * ratio).int(), torch.ceil(bbox[:, 3] * ratio).int()
    # ys, xs = ys * ratio, xs * ratio
    # ys_floor, xs_floor = ys.floor().long(), xs.floor().long()
    # hs, ws = hs / in_h, ws / in_w

    target = torch.zeros((out_h, out_w, 5 * n_bbox_predict + n_class))
    target[:, :, 5 * n_bbox_predict] = 1

    for i in range(n_gt):
        bbox = bboxes[i]
        # y, x = ys[i], xs[i]
        # h, w = hs[i], ws[i]
        h, w = (bbox[2] - bbox[0]) / in_h, (bbox[3] - bbox[1]) / in_w
        y, x = (bbox[0] + .5 * h) * ratio, (bbox[1] + .5 * w) * ratio

        y_cell_idx, x_cell_idx = int(y), int(x)
        y_cell, x_cell = y - int(y), x - int(x)
        class_ = classes[i]

        for j in range(2):
            target[y_cell_idx, x_cell_idx, 5 * j] = x_cell
            target[y_cell_idx, x_cell_idx, 5 * j + 1] = y_cell
            target[y_cell_idx, x_cell_idx, 5 * j + 2] = w
            target[y_cell_idx, x_cell_idx, 5 * j + 3] = h
            target[y_cell_idx, x_cell_idx, 5 * j + 4] = 1

        target[y_cell_idx, x_cell_idx, 5 * n_bbox_predict + class_] = 1
        target[y_cell_idx, x_cell_idx, 5 * n_bbox_predict] = 0


    # target[:, :, 5 * n_bbox] = torch.ones(out_size)
    # for i in range(n_bbox):
    #     target[bbox_y1_warp[i]:bbox_y2_warp[i] + 1, bbox_x1_warp[i]:bbox_x2_warp[i] + 1, 5 * i:5 * i + 4] = \
    #         torch.Tensor([bbox_y[i], bbox_x[i] ,bbox_h[i], bbox_w[i]]).expand((bbox_h_warp, bbox_w_warp, 4))
    #     target[bbox_y1_warp[i]:bbox_y2_warp[i] + 1, bbox_x1_warp[i]:bbox_x2_warp[i] + 1, 5 * i + 4] = torch.ones((bbox_h_warp[i], bbox_w_warp[i]))
    #     target[bbox_y1_warp[i]:bbox_y2_warp[i] + 1, bbox_x1_warp[i]:bbox_x2_warp[i] + 1, 5 * n_bbox + class_.item() - 1] = torch.ones((bbox_h_warp[i], bbox_w_warp[i]))

    return target


def make_batch(data_list):
    n_data = len(data_list)
    data_batch = data_list[0].unsqueeze(0)

    for i in range(1, n_data):
        temp = data_list[i].unsqueeze(0)
        data_batch = torch.cat([data_batch, temp], dim=0)

    return data_batch


def calculate_precision(output, ground_truths):
    n_batch = output.shape[0]




def calculate_iou(box1, box2):
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    y1 = np.maximum(box1[0], box2[0])
    x1 = np.maximum(box1[1], box2[1])
    y2 = np.minimum(box1[2], box2[2])
    x2 = np.minimum(box1[3], box2[3])

    iou = 0
    if y1 < y2 and x1 < x2:
        inter = (y2 - y1) * (x2 - x1)
        union = area1 + area2 - inter
        iou = inter / union

    return iou