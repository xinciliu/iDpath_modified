import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_withlogits_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)

def sce_withlogits_loss(output, target):
    return F.softmax_cross_entropy_with_logits(output, target)