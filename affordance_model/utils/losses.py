import torch
import torch.nn.functional as F


def compute_mIoU(logits, gt, threshold=0.5):
    if logits.shape[1] == 1 or len(logits.shape) == 3:
        if threshold == 0.5:
            pred = logits.round().byte()
        else:
            pred = logits > threshold
    else:
        pred = logits.argmax(dim=1).byte()
    intersection = ((pred == 1) & (gt == 1)).sum().float()
    union = ((pred == 1) | (gt == 1)).sum().float()
    return intersection/(union+1.)


def pixel2spatial(xs, H, W):
    """Converts a tensor of coordinates to a boolean spatial map. 
    """
    xs_spatial = []
    for x in xs:
        pos = x[x[:, 2] == 1][:, :2]
        pos_label = torch.zeros(1, 1, H, W)
        pos_label[:, :, pos[:, 0], pos[:, 1]] = 1
        xs_spatial.append(pos_label)
    xs_spatial = torch.cat(xs_spatial, dim=0).long()
    return xs_spatial


# https://github.com/kevinzakka/form2fit/blob/099a4ceac0ec60f5fbbad4af591c24f3fff8fa9e/form2fit/code/ml/losses.py#L305
def compute_dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)
