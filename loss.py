import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def yolo_pretrain_custom_loss(predict, target):
    losses = -1 * (target * torch.log(predict + 1e-15) + (1 - target) * torch.log(1 - predict + 1e-15))
    batch = losses.shape[0]
    loss = losses.sum() / batch

    return loss


def yolo_custom_loss(predict, target, n_bbox, lambda_coord, lambda_noobj):
    coord_losses = torch.zeros(target.shape[1:3]).to(device)
    obj_losses = torch.zeros(target.shape[1:3]).to(device)
    class_losses = torch.zeros(target.shape[1:3]).to(device)

    n_batch = predict.shape[0]
    for b in range(n_batch):
        is_obj_global = torch.zeros(target.shape[1:3]).to(device)
        for i in range(n_bbox):
            coord_losses += torch.square(predict[b, :, :, 5 * i] - target[b, :, :, 5 * i])
            coord_losses += torch.square(predict[b, :, :, 5 * i + 1] - target[b, :, :, 5 * i + 1])
            coord_losses += torch.square(torch.sqrt(predict[b, :, :, 5 * i + 2]) - torch.sqrt(target[b, :, :, 5 * i + 2]))
            coord_losses += torch.square(torch.sqrt(predict[b, :, :, 5 * i + 3]) - torch.sqrt(target[b, :, :, 5 * i + 3]))

            is_obj = target[b, :, :, 5 * i + 4]
            if b == 0:
                is_obj_global += is_obj
            obj_losses_temp = torch.square(predict[b, :, :, 5 * i + 4] - is_obj)
            obj_losses += (is_obj + (1 - is_obj) * lambda_noobj) * obj_losses_temp

        if b == 0:
            is_obj_global /= n_batch

        class_losses_temp = torch.square(predict[b, :, :, 5 * n_bbox:] - target[b, :, :, 5 * n_bbox:])
        class_losses_temp = class_losses_temp.sum(dim=2)
        class_losses += class_losses_temp * is_obj_global

    coord_loss = lambda_coord * coord_losses.sum() / n_batch
    obj_loss = obj_losses.sum() / n_batch
    class_loss = class_losses.sum() / n_batch

    loss = coord_loss + obj_loss + class_loss

    print(coord_loss, obj_loss, class_loss)

    return loss


if __name__ == '__main__':
    a = torch.Tensor((2, 20))
