import torch
from torch import Tensor


def dice_coeff(
    input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6
) -> Tensor:
    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter
        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(
    input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6
) -> Tensor:
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(
            input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon
        )
    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


class DiceLoss(torch.nn.Module):

    def __init__(self, multiclass: bool = True) -> None:
        super().__init__()
        self.multiclass = multiclass

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        return dice_loss(outputs, targets, multiclass=self.multiclass)