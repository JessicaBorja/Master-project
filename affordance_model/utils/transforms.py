import torch

class ScaleImageTensor(object):
    """Scale tensor of shape (batch, C, H, W) containing images to [0, 255] range
    Args:
        tensor (torch.tensor): Tensor to be scaled.
    Returns:
        Tensor: Scaled tensor.
    """
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        return tensor.float().div(255)

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = torch.Tensor(std)
        self.mean = torch.Tensor(mean)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)

class ThresholdMasks(object):
    def __init__(self, threshold):
        # Mask is between 0 and 255
        self.threshold = threshold

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        return (tensor > self.threshold).long()