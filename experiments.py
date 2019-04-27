import torch
from network import ModLayer

# a = torch.randn(2, 2)
# a = ((a * 3) / (a - 1))
#
# print(a.requires_grad)
# c = (a * a).sum()
# print(c)
#
# a.requires_grad_(True)
# print(a.requires_grad)
# b = torch.sin(a * a)
# print(b, b.grad_fn)

layer = ModLayer(5)
a = torch.arange(0, 10).view(-1, 1)
print(layer(a))

