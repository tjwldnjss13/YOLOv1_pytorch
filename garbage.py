import torch


a = torch.Tensor([1, 2])
# a = a.unsqueeze(0).unsqueeze(0)
a = a.expand((3, 3, 3))
print(a)