# -*- coding: utf-8 -*-
# Loss functions (PyTorch and own defined)
#
# Own defined loss functions:
# xentropy_loss, dice_loss, mse_loss and msge_loss (https://github.com/vqdang/hover_net)
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


import torch
import torch.nn.functional as F
from typing import List, Tuple
from torch import nn
from torch.nn.modules.loss import _Loss


class XentropyLoss(_Loss):
    """Cross entropy loss"""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(size_average=None, reduce=None, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Assumes NHWC shape of array, must be torch.float32 dtype

        Args:
            input (torch.Tensor): Ground truth array with shape (N, H, W, C) with N being the batch-size, H the height, W the width and C the number of classes
            target (torch.Tensor): Prediction array with shape (N, H, W, C) with N being the batch-size, H the height, W the width and C the number of classes

        Returns:
            torch.Tensor: Cross entropy loss, with shape () [scalar], grad_fn = MeanBackward0
        """

        epsilon = 10e-8
        # scale preds so that the class probs of each sample sum to 1
        pred = input / torch.sum(input, -1, keepdim=True)
        # manual computation of crossentropy
        pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
        loss = -torch.sum((target * torch.log(pred)), -1, keepdim=True)
        loss = loss.mean() if self.reduction == "mean" else loss.sum()

        return loss


class DiceLoss(_Loss):
    """Dice loss

    Args:
        smooth (float, optional): Smoothing value. Defaults to 1e-3.
    """

    def __init__(self, smooth: float = 1e-3) -> None:
        super().__init__(size_average=None, reduce=None, reduction="mean")
        self.smooth = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Assumes NHWC shape of array, must be torch.float32 dtype

        `pred` and `true` must be of torch.float32. Assuming of shape NxHxWxC.

        Args:
            input (torch.Tensor): Prediction array with shape (N, H, W, C) with N being the batch-size, H the height, W the width and C the number of classes
            target (torch.Tensor): Ground truth array with shape (N, H, W, C) with N being the batch-size, H the height, W the width and C the number of classes

        Returns:
            torch.Tensor: Dice loss, with shape () [scalar], grad_fn=SumBackward0
        """
        inse = torch.sum(input * target, (0, 1, 2))
        l = torch.sum(input, (0, 1, 2))
        r = torch.sum(target, (0, 1, 2))
        loss = 1.0 - (2.0 * inse + self.smooth) / (l + r + self.smooth)
        loss = torch.sum(loss)

        return loss


class MSELossMaps(_Loss):
    """Calculate mean squared error loss for combined horizontal and vertical maps of segmentation tasks."""

    def __init__(self) -> None:
        super().__init__(size_average=None, reduce=None, reduction="mean")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss calculation

        Args:
            input (torch.Tensor): Prediction of combined horizontal and vertical maps
                with shape (N, H, W, 2), channel 0 is vertical and channel 1 is horizontal
            target (torch.Tensor): Ground truth of combined horizontal and vertical maps
                with shape (N, H, W, 2), channel 0 is vertical and channel 1 is horizontal

        Returns:
            torch.Tensor: Mean squared error per pixel with shape (N, H, W, 2), grad_fn=SubBackward0

        """
        loss = input - target
        loss = (loss * loss).mean()
        return loss


class MSGELossMaps(_Loss):
    def __init__(self) -> None:
        super().__init__(size_average=None, reduce=None, reduction="mean")

    def get_sobel_kernel(
        self, size: int, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sobel kernel with a given size.

        Args:
            size (int): Kernel site
            device (str): Cuda device

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Horizontal and vertical sobel kernel, each with shape (size, size)
        """
        assert size % 2 == 1, "Must be odd, get size=%d" % size

        h_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device=device,
            requires_grad=False,
        )
        v_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device=device,
            requires_grad=False,
        )
        h, v = torch.meshgrid(h_range, v_range, indexing="ij")
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        return kernel_h, kernel_v

    def get_gradient_hv(self, hv: torch.Tensor, device: str) -> torch.Tensor:
        """For calculating gradient of horizontal and vertical prediction map


        Args:
            hv (torch.Tensor): horizontal and vertical map
            device (str): CUDA device

        Returns:
            torch.Tensor: Gradient with same shape as input
        """
        kernel_h, kernel_v = self.get_sobel_kernel(5, device=device)
        kernel_h = kernel_h.view(1, 1, 5, 5)  # constant
        kernel_v = kernel_v.view(1, 1, 5, 5)  # constant

        h_ch = hv[..., 0].unsqueeze(1)  # Nx1xHxW
        v_ch = hv[..., 1].unsqueeze(1)  # Nx1xHxW

        # can only apply in NCHW mode
        h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
        v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
        dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
        dhv = dhv.permute(0, 2, 3, 1).contiguous()  # to NHWC
        return dhv

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        focus: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        focus = (focus[..., None]).float()  # assume input NHW
        focus = torch.cat([focus, focus], axis=-1).to(device)
        true_grad = self.get_gradient_hv(target, device)
        pred_grad = self.get_gradient_hv(input, device)
        loss = pred_grad - true_grad
        loss = focus * (loss * loss)
        # artificial reduce_mean with focused region
        loss = loss.sum() / (focus.sum() + 1.0e-8)
        return loss


class FocalTverskyLoss(nn.Module):
    """FocalTverskyLoss

    PyTorch implementation of the Focal Tversky Loss Function for multiple classes
    doi: 10.1109/ISBI.2019.8759329
    Abraham, N., & Khan, N. M. (2019).
    A Novel Focal Tversky Loss Function With Improved Attention U-Net for Lesion Segmentation.
    In International Symposium on Biomedical Imaging. https://doi.org/10.1109/isbi.2019.8759329

    @ Fabian Hörst, fabian.hoerst@uk-essen.de
    Institute for Artifical Intelligence in Medicine,
    University Medicine Essen

    Args:
        alpha_t (float, optional): Alpha parameter for tversky loss (multiplied with false-negatives). Defaults to 0.7.
        beta_t (float, optional): Beta parameter for tversky loss (multiplied with false-positives). Defaults to 0.3.
        gamma_f (float, optional): Gamma Focal parameter. Defaults to 4/3.
        smooth (float, optional): Smooting factor. Defaults to 0.000001.
    """

    def __init__(
        self,
        alpha_t: float = 0.7,
        beta_t: float = 0.3,
        gamma_f: float = 4 / 3,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha_t = alpha_t
        self.beta_t = beta_t
        self.gamma_f = gamma_f
        self.smooth = smooth
        self.num_classes = 2

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss calculation

        Args:
            input (torch.Tensor): Predictions, logits (without Softmax). Shape: (batch-size, H, W, num_classes)
            target (torch.Tensor): Targets, either flattened (Shape: (batch.size, H, W) or as one-hot encoded (Shape: (batch-size, H, W, num_classes)).

        Raises:
            ValueError: Error if there is a shape missmatch

        Returns:
            torch.Tensor: FocalTverskyLoss (weighted)
        """
        if input.shape[-1] != self.num_classes:
            raise ValueError(
                "Predictions must be a logit tensor with the last dimension shape beeing equal to the number of classes"
            )
        if len(target.shape) != len(input.shape):
            # convert the targets to onehot
            target = F.one_hot(target, num_classes=self.num_classes)

        # flatten
        target = target.view(-1)
        input = torch.softmax(input, dim=-1).view(-1)

        # calculate true positives, false positives and false negatives
        tp = (input * target).sum()
        fp = ((1 - target) * input).sum()
        fn = (target * (1 - input)).sum()

        Tversky = (tp + self.smooth) / (
            tp + self.alpha_t * fn + self.beta_t * fp + self.smooth
        )
        FocalTversky = (1 - Tversky) ** self.gamma_f

        return FocalTversky


class MCFocalTverskyLoss(FocalTverskyLoss):
    """Multiclass FocalTverskyLoss

    PyTorch implementation of the Focal Tversky Loss Function for multiple classes
    doi: 10.1109/ISBI.2019.8759329
    Abraham, N., & Khan, N. M. (2019).
    A Novel Focal Tversky Loss Function With Improved Attention U-Net for Lesion Segmentation.
    In International Symposium on Biomedical Imaging. https://doi.org/10.1109/isbi.2019.8759329

    @ Fabian Hörst, fabian.hoerst@uk-essen.de
    Institute for Artifical Intelligence in Medicine,
    University Medicine Essen

    Args:
        alpha_t (float, optional): Alpha parameter for tversky loss (multiplied with false-negatives). Defaults to 0.7.
        beta_t (float, optional): Beta parameter for tversky loss (multiplied with false-positives). Defaults to 0.3.
        gamma_f (float, optional): Gamma Focal parameter. Defaults to 4/3.
        smooth (float, optional): Smooting factor. Defaults to 0.000001.
        num_classes (int, optional): Number of output classes. For binary segmentation, prefer FocalTverskyLoss (speed optimized). Defaults to 2.
        class_weights (List[int], optional): Weights for each class. If not provided, equal weight. Length must be equal to num_classes. Defaults to None.
    """

    def __init__(
        self,
        alpha_t: float = 0.7,
        beta_t: float = 0.3,
        gamma_f: float = 4 / 3,
        smooth: float = 0.000001,
        num_classes: int = 2,
        class_weights: List[int] = None,
    ) -> None:
        super().__init__(alpha_t, beta_t, gamma_f, smooth)
        self.num_classes = num_classes
        if class_weights is None:
            self.class_weights = [1 for i in range(self.num_classes)]
        else:
            assert (
                len(class_weights) == self.num_classes
            ), "Please provide matching weights"
            self.class_weights = class_weights
        self.class_weights = torch.Tensor(self.class_weights)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss calculation

        Args:
            input (torch.Tensor): Predictions, logits (without Softmax). Shape: (batch-size, H, W, num_classes)
            target (torch.Tensor): Targets, either flattened (Shape: (batch.size, H, W) or as one-hot encoded (Shape: (batch-size, H, W, num_classes)).

        Raises:
            ValueError: Error if there is a shape missmatch

        Returns:
            torch.Tensor: FocalTverskyLoss (weighted)
        """
        if input.shape[-1] != self.num_classes:
            raise ValueError(
                "Predictions must be a logit tensor with the last dimension shape beeing equal to the number of classes"
            )
        if len(target.shape) != len(input.shape):
            # convert the targets to onehot
            target = F.one_hot(target, num_classes=self.num_classes)

        # Softmax
        input = torch.softmax(input, dim=-1)
        # Reshape
        input = torch.permute(input, (3, 1, 2, 0))
        target = torch.permute(target, (3, 1, 2, 0))

        input = torch.flatten(input, start_dim=1)
        target = torch.flatten(target, start_dim=1)

        tp = torch.sum(input * target, 1)
        fp = torch.sum((1 - target) * input, 1)
        fn = torch.sum(target * (1 - input), 1)

        Tversky = (tp + self.smooth) / (
            tp + self.alpha_t * fn + self.beta_t * fp + self.smooth
        )
        FocalTversky = (1 - Tversky) ** self.gamma_f

        self.class_weights = self.class_weights.to(FocalTversky.device)
        return torch.sum(self.class_weights * FocalTversky)


def retrieve_loss_fn(loss_name: dict, **kwargs) -> _Loss:
    """Return the loss function with given name defined in the LOSS_DICT and initialize with kwargs

    kwargs must match with the parameters defined in the initialization method of the selected loss object

    Args:
        loss_name (dict): Name of the loss function

    Returns:
        _Loss: Loss
    """
    loss_fn = LOSS_DICT[loss_name]
    loss_fn = loss_fn(**kwargs)

    return loss_fn


LOSS_DICT = {
    "xentropy_loss": XentropyLoss,
    "dice_loss": DiceLoss,
    "mse_loss_maps": MSELossMaps,
    "msge_loss_maps": MSGELossMaps,
    "FocalTverskyLoss": FocalTverskyLoss,
    "MCFocalTverskyLoss": MCFocalTverskyLoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "L1Loss": nn.L1Loss,
    "MSELoss": nn.MSELoss,
    "CTCLoss": nn.CTCLoss,
    "NLLLoss": nn.NLLLoss,
    "PoissonNLLLoss": nn.PoissonNLLLoss,
    "GaussianNLLLoss": nn.GaussianNLLLoss,
    "KLDivLoss": nn.KLDivLoss,
    "BCELoss": nn.BCELoss,
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    "MarginRankingLoss": nn.MarginRankingLoss,
    "HingeEmbeddingLoss": nn.HingeEmbeddingLoss,
    "MultiLabelMarginLoss": nn.MultiLabelMarginLoss,
    "HuberLoss": nn.HuberLoss,
    "SmoothL1Loss": nn.SmoothL1Loss,
    "SoftMarginLoss": nn.SoftMarginLoss,
    "MultiLabelSoftMarginLoss": nn.MultiLabelSoftMarginLoss,
    "CosineEmbeddingLoss": nn.CosineEmbeddingLoss,
    "MultiMarginLoss": nn.MultiMarginLoss,
    "TripletMarginLoss": nn.TripletMarginLoss,
    "TripletMarginWithDistanceLoss": nn.TripletMarginWithDistanceLoss,
}
