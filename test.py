# %%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from data_loader.data_loader import get_dataloader
from model.model import UNet
import yaml
import torch

with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = UNet(21).to(device)

model.load_state_dict(torch.load("check_point/model_0.pth"))
model.eval()

train_loader, test_loader = get_dataloader("data", 1)
# %%

for module in model.named_modules():
    print(module)
    print("-----------")

# %%
count = torch.zeros(21, dtype=torch.int64)
print(count)

for img, label in iter(train_loader):
    idx, c = torch.unique(label, return_counts=True)
    for i, n in zip(idx, c):
        count[int(i)] += n

print(count.shape)
print(count)
print(count.sum() / count)
# %%
img, masks = next(iter(train_loader))

print(img.shape)
print(masks.shape)
# print(masks[0].long().dtype)


def inverse_norm(img):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    result = img
    for i in range(3):
        result[:, i, ...] = img[:, i, ...] * std[i] + mean[i]

    return result


images = inverse_norm(img)
image = images[0].permute(1, 2, 0).numpy()
mask = masks[0].numpy()

# %%
pred = model(img.to(device))
pred_idx = torch.argmax(pred, dim=1)
print(torch.unique(pred_idx[0], return_counts=True))

print(np.unique(mask))
print(mask.shape)

fig, ax = plt.subplots(figsize=(6, 12), nrows=1, ncols=3)

for a in ax:
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)

ax[0].imshow(image)
ax[1].imshow(mask)
ax[2].imshow(pred_idx[0].detach().cpu().numpy())

# %%
print(
    torch.sum(pred_idx[1].detach().cpu() == torch.tensor(mask)).item()
    / torch.tensor(image).size(0) ** 2
)
# %%
import torch
from torch import nn

x = torch.tensor(
    [
        [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
        [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
        [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
    ],
    dtype=torch.float64,
)

y = torch.tensor(
    [
        [[1, 0, 1], [1, 0, 1], [1, 0, 1]],
        [[0, 1, 1], [0, 1, 1], [0, 1, 1]],
        [[1, 1, 0], [1, 1, 0], [1, 1, 0]],
    ],
    dtype=torch.float64,
)

loss = nn.BCEWithLogitsLoss()
out = loss(x, y)
print(x)
print(y)
print(out.shape)
print(out)

# %%
from torch.nn.functional import one_hot
import torch

x = torch.tensor(
    [[[0, 0, 0], [0, 3, 0], [0, 0, 0]], [[0, 0, 0], [0, 2, 0], [0, 0, 0]]],
    dtype=torch.int64,
)

print(x.shape)
print(one_hot(x, 5).shape)
print(one_hot(x, 5).permute(0, 3, 1, 2))
