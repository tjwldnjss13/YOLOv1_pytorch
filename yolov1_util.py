import torch
import numpy as np

from utils import make_batch


def calculate_iou(box1, box2):
    # Inputs:
    #    box1, box2: [cx, cy, w, h] tensor

    y1_box1 = box1[1] - .5 * box1[3]
    x1_box1 = box1[0] - .5 * box1[2]
    y2_box1 = y1_box1 + box1[3]
    x2_box1 = x1_box1 + box1[2]

    y1_box2 = box2[1] - .5 * box2[3]
    x1_box2 = box2[0] - .5 * box2[2]
    y2_box2 = y1_box2 + box2[3]
    x2_box2 = x1_box2 + box2[2]

    y1_inter = torch.max(y1_box1, y1_box2)
    x1_inter = torch.max(x1_box1, x1_box2)
    y2_inter = torch.min(y2_box1, y2_box2)
    x2_inter = torch.min(x2_box1, x2_box2)

    area_inter = (y2_inter - y1_inter) * (x2_inter - x1_inter)

    area_box1 = box1[2] * box1[3]
    area_box2 = box2[2] * box2[3]
    area_union = area_box1 + area_box2 - area_inter

    iou = area_inter / area_union

    return iou


# a = torch.Tensor([2.5, 2.5, 5, 5])
# b = torch.Tensor([4.5, 4.5, 5, 5])
# iou = calculate_iou(a, b)
# print(iou)


def generate_target_batch(annotation, n_bbox_predict, n_class, in_size, out_size):
    n_ann = len(annotation)

    target_tensors = []
    for i in range(n_ann):
        ann = annotation[i]
        classes, bboxes, filename = ann['class'], ann['bbox'], ann['filename']

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


def get_infos_from_output(image, output):
    # Inputs:
    #    image: PIL Image
    #    output: (7, 7, 31) tensor
    # Outputs:
    #    box_dict
    #    prob_dict

    img = np.array(image)

    h_img, w_img, _ = img.shape
    h_img_grid, w_img_grid = int(h_img / 7), int(w_img / 7)

    box_dict = {}
    prob_dict = {}

    for i in range(7):
        for j in range(7):
            cls = torch.argmax(output[i, j, 10:])

            # Background는 무시
            if cls > 0:
                boxes = []
                probs = []

                for b in range(2):
                    box_cell = output[i, j]

                    x = (i + box_cell[5 * b].item()) * w_img_grid
                    y = (j + box_cell[5 * b + 1].item()) * h_img_grid
                    w = output[i, j, 5 * b + 2].item() * w_img
                    h = box_cell[5 * b + 3].item() * h_img

                    x_min = int(x - .5 * w)
                    y_min = int(y - .5 * h)
                    x_max = int(x + .5 * w)
                    y_max = int(y + .5 * h)

                    box = [y_min, x_min, y_max, x_max]
                    boxes.append(box)

                    probs.append(box_cell[b + 4].item())

                if cls not in box_dict:
                    box_dict[cls] = boxes
                    prob_dict[cls] = probs
                else:
                    box_dict[cls] += boxes
                    prob_dict[cls] += probs

    return box_dict, prob_dict


def mean_average_precision(output, ground_truths):
    n_batch = output.shape[0]


def calculate_average_precision():
    pass


def calculate_precision(n_true_positive, n_false_positive):
    return n_true_positive / (n_true_positive + n_false_positive)


def calculate_recall(n_true_positive, n_false_negative):
    return n_true_positive / (n_true_positive + n_false_negative)


# def calculate_iou(box1, box2):
#     area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
#
#     y1 = np.maximum(box1[0], box2[0])
#     x1 = np.maximum(box1[1], box2[1])
#     y2 = np.minimum(box1[2], box2[2])
#     x2 = np.minimum(box1[3], box2[3])
#
#     iou = 0
#     if y1 < y2 and x1 < x2:
#         inter = (y2 - y1) * (x2 - x1)
#         union = area1 + area2 - inter
#         iou = inter / union
#
#     return iou

