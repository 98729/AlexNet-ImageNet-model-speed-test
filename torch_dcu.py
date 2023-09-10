import torch

device = torch.device('cuda')
print(torch.cuda.device_count())
print(torch.cuda.is_available())
a = torch.ones(2, 2).to(device)
torch.svd(a)
