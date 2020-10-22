import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def yolo_custom_loss(predict, target, n_bbox, lambda_coord, lambda_noobj):
    coord_losses = torch.zeros(target.shape[:2]).to(device)
    obj_losses = torch.zeros(target.shape[:2]).to(device)
    class_losses = torch.zeros(target.shape[:2]).to(device)

    for b in range(predict.shape[0]):
        is_obj_global = torch.zeros(target.shape[:2]).to(device)
        for i in range(n_bbox):
            coord_losses += torch.square(predict[b, :, :, 5 * i] - target[:, :, 5 * i])
            coord_losses += torch.square(predict[b, :, :, 5 * i + 1] - target[:, :, 5 * i + 1])
            coord_losses += torch.square(torch.sqrt(predict[b, :, :, 5 * i + 2]) - torch.sqrt(target[:, :, 5 * i + 2]))
            coord_losses += torch.square(torch.sqrt(predict[b, :, :, 5 * i + 3]) - torch.sqrt(target[:, :, 5 * i + 3]))

            is_obj = target[:, :, 5 * i + 4]
            if b == 0:
                is_obj_global += is_obj
            obj_losses_temp = torch.square(predict[b, :, :, 5 * i + 4] - is_obj)
            obj_losses += (is_obj + (1 - is_obj) * lambda_noobj) * obj_losses_temp

        if b == 0:
            is_obj_global /= predict.shape[0]

        class_losses_temp = torch.square(predict[b, :, :, 5 * n_bbox:] - target[:, :, 5 * n_bbox:])
        class_losses_temp = class_losses_temp.sum(dim=2)
        class_losses += class_losses_temp * is_obj_global

    coord_loss = lambda_coord * coord_losses.sum() / predict.shape[0]
    obj_loss = obj_losses.sum() / predict.shape[0]
    class_loss = class_losses.sum() / predict.shape[0]

    loss = coord_loss + obj_loss + class_loss

    return loss