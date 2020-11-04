import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def yolo_pretrain_target_generator(labels):
    pass


def target_generator(bbox, class_, n_bbox, n_class, in_size, out_size):
    in_h, in_w = in_size[0], in_size[1]
    out_h, out_w = out_size[0], out_size[1]
    bbox_h, bbox_w = bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1]
    bbox_y, bbox_x = bbox[:, 0] + .5 * bbox_h, bbox[:, 1] + .5 * bbox_w

    objs = torch.zeros(out_size, dtype=torch.long).to(device)
    ratio = out_h / in_h
    bbox_y1_warp, bbox_x1_warp = torch.floor(bbox[:, 0] * ratio).int(), torch.floor(bbox[:, 1] * ratio).int()
    bbox_y2_warp, bbox_x2_warp = torch.ceil(bbox[:, 2] * ratio).int(), torch.ceil(bbox[:, 3] * ratio).int()
    bbox_h_warp = bbox_y2_warp - bbox_y1_warp
    bbox_w_warp = bbox_x2_warp - bbox_x1_warp

    for i in range(n_bbox):
        objs[bbox_y1_warp[i]:bbox_y2_warp[i] + 1, bbox_x1_warp[i]:bbox_x2_warp[i] + 1] = torch.ones((bbox_h_warp, bbox_w_warp))

    target = torch.zeros((out_h, out_w, 5 * n_bbox + n_class)).to(device)
    target[:, :, 5 * n_bbox] = torch.ones(out_size)
    for i in range(n_bbox):
        # for m in range(bbox_y1_warp[i], bbox_y2_warp[i] + 1):
        #     for n in range(bbox_x1_warp[i], bbox_x2_warp[i] + 1):
        #         target[m, n, 5 * i:5 * i + 4] = torch.Tensor([bbox_y[i], bbox_x[i], bbox_h[i], bbox_w[i]])
        #         target[m, n, 5 * i + 4] = objs
        target[bbox_y1_warp[i]:bbox_y2_warp[i] + 1, bbox_x1_warp[i]:bbox_x2_warp[i] + 1, 5 * i:5 * i + 4] = \
            torch.Tensor([bbox_y[i], bbox_x[i] ,bbox_h[i], bbox_w[i]]).expand((bbox_h_warp, bbox_w_warp, 4))
        for j in range(4):
            target[:, :, 5 * i + j] *= objs

    for y_ in range(out_h):
        for x_ in range(out_w):
            for c in class_:
                target[y_, x_, 5 * n_bbox + c * objs[y_, x_]] = 1

    return target
