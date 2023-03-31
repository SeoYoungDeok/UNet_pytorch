import torch.nn.functional as F


def binary_cross_entropy_with_logits(pred, label):
    return F.binary_cross_entropy_with_logits(pred, label)
