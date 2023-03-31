# %%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from data_loader.data_loader import get_dataloader

train_loader, test_loader = get_dataloader("data", 8)

# %%
img, masks = next(iter(train_loader))

# %%
print(img.shape)
print(masks.shape)
print(masks[0].long().dtype)
# %%


def inverse_norm(img):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    result = img
    for i in range(3):
        result[:, i, ...] = img[:, i, ...] * std[i] + mean[i]

    return result


images = inverse_norm(img)

# %%
image = images[6].permute(1, 2, 0).numpy()
mask = masks[6].numpy()

fig, ax = plt.subplots(figsize=(6, 12), nrows=1, ncols=2)
ax[0].imshow(image)
ax[1].imshow(mask)
