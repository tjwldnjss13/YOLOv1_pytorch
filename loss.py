import torch

from yolov1_util import calculate_iou

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def yolo_pretrain_custom_loss(predict, target):
    losses = -1 * (target * torch.log(predict + 1e-15) + (1 - target) * torch.log(1 - predict + 1e-15))
    batch = losses.shape[0]
    loss = losses.sum() / batch

    return loss


def yolo_custom_loss(predict, target, n_bbox_predict, lambda_coord=5, lambda_noobj=.5):
    coord_losses = torch.zeros(7, 7, 2).to(device)
    obj_losses = torch.zeros(target.shape[1:3]).to(device)
    class_losses = torch.zeros(target.shape[1:3]).to(device)

    coord_loss = torch.zeros(1).to(device)

    n_batch = predict.shape[0]
    for b in range(n_batch):
        is_obj_global = torch.zeros(target.shape[1:3]).to(device)

        obj_responsible_mask = torch.zeros(7, 7, 2).to(device)
        noobj_responsible_mask = torch.zeros(7, 7, 2).to(device)

        # Find responsible box
        for i in range(n_bbox_predict):
            obj_responsible_mask[:, :, i] = target[b, :, :, 5 * i + 4]
            noobj_responsible_mask[:, :, i] = target[b, :, :, 5 * i + 4]

        for s1 in range(7):
            for s2 in range(7):
                if obj_responsible_mask[s1, s2, 0] == 1:
                    box1 = predict[b, s1, s2, :4]
                    box2 = predict[b, s1, s2, 5:9]
                    gt = target[b, s1, s2, :4]

                    iou1 = calculate_iou(box1, gt)
                    iou2 = calculate_iou(box2, gt)

                    if iou1 > iou2:
                        obj_responsible_mask[s1, s2, 0] = 0
                    else:
                        obj_responsible_mask[s1, s2, 1] = 0

        # Calculate coordinates loss
        for i in range(n_bbox_predict):
            coord_losses = torch.square(predict[b, :, :, 5 * i] - target[b, :, :, 5 * i]) \
                            + torch.square(predict[b, :, :, 5 * i + 1] - target[b, :, :, 5 * i + 1])
            coord_losses *= obj_responsible_mask[:, :, i]

            # coord_losses += torch.square(predict[b, :, :, 5 * i] - target[b, :, :, 5 * i])
            # coord_losses += torch.square(predict[b, :, :, 5 * i + 1] - target[b, :, :, 5 * i + 1])
            # coord_losses += torch.square(torch.sqrt(predict[b, :, :, 5 * i + 2]) - torch.sqrt(target[b, :, :, 5 * i + 2]))
            # coord_losses += torch.square(torch.sqrt(predict[b, :, :, 5 * i + 3]) - torch.sqrt(target[b, :, :, 5 * i + 3]))

            is_obj = target[b, :, :, 5 * i + 4]
            if b == 0:
                is_obj_global += is_obj
            obj_losses_temp = torch.square(predict[b, :, :, 5 * i + 4] - is_obj)
            obj_losses += (is_obj + (1 - is_obj) * lambda_noobj) * obj_losses_temp

        if b == 0:
            is_obj_global /= n_batch

        coord_loss += coord_losses.sum()

        class_losses_temp = torch.square(predict[b, :, :, 5 * n_bbox_predict:] - target[b, :, :, 5 * n_bbox_predict:])
        class_losses_temp = class_losses_temp.sum(dim=2)
        class_losses += class_losses_temp * is_obj_global

    coord_loss = lambda_coord * coord_losses.sum() / n_batch
    obj_loss = obj_losses.sum() / n_batch
    class_loss = class_losses.sum() / n_batch

    loss = coord_loss + obj_loss + class_loss

    # print(coord_loss.item(), obj_loss.item(), class_loss.item(), end=' ')

    return loss


if __name__ == '__main__':
    a = torch.Tensor((2, 20))
