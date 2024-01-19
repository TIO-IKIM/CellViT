# -*- coding: utf-8 -*-
# Loss functions (PyTorch and own defined)
#
# Own defined loss functions:
# xentropy_loss, dice_loss, mse_loss and msge_loss (https://github.com/vqdang/hover_net)
# WeightedBaseLoss, MAEWeighted, MSEWeighted, BCEWeighted, CEWeighted (https://github.com/okunator/cellseg_models.pytorch)
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


import torch
import torch.nn.functional as F
from typing import List, Tuple
from torch import nn
from torch.nn.modules.loss import _Loss
from base_ml.base_utils import filter2D, gaussian_kernel2d


class XentropyLoss(_Loss):
    """Cross entropy loss.

    This class represents the cross entropy loss function for multi-class classification tasks.

    Args:
        reduction (str, optional): The reduction method for the loss. Defaults to 'mean'.

    Attributes:
        reduction (str): The reduction method for the loss. Possible values are 'mean' and 'sum'.

    Methods:
        __init__(reduction: str = "mean") -> None:
            Initializes the XentropyLoss instance.
        forward(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            Computes the cross entropy loss

    Returns:
        torch.Tensor: The computed cross entropy loss.

    Examples:
        >>> loss_fn = XentropyLoss()
        >>> input = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]])
        >>> target = torch.tensor([[[[0, 1], [1, 0]], [[1, 0], [0, 1]]]])
        >>> loss = loss_fn(input, target)
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(size_average=None, reduce=None, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the cross entropy loss.

        Args:
            input (torch.Tensor): Ground truth array with shape (N, C, H, W) where N is the batch size,
                C is the number of classes, H is the height, and W is the width.
            target (torch.Tensor): Prediction array with shape (N, C, H, W) where N is the batch size,
                C is the number of classes, H is the height, and W is the width.

        Returns:
            torch.Tensor: The computed cross entropy loss.

        Raises:
            AssertionError: If the input and target tensors have different shapes.

        Examples:
            >>> loss_fn = XentropyLoss()
            >>> input = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]])
            >>> target = torch.tensor([[[[0, 1], [1, 0]], [[1, 0], [0, 1]]]])
            >>> loss = loss_fn(input, target)
        """
        assert (
            input.shape == target.shape
        ), "Input and target tensors must have the same shape"

        # reshape
        input = input.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)

        epsilon = 10e-8
        # scale preds so that the class probs of each sample sum to 1
        pred = input / torch.sum(input, -1, keepdim=True)
        # manual computation of crossentropy
        pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
        loss = -torch.sum((target * torch.log(pred)), -1, keepdim=True)
        loss = loss.mean() if self.reduction == "mean" else loss.sum()

        return loss


class DiceLoss(_Loss):
    """Dice loss.

    This class represents the Dice loss function for segmentation tasks.

    Args:
        smooth (float, optional): Smoothing value. Defaults to 1e-3.

    Attributes:
        smooth (float): Smoothing value.

    Methods:
        __init__(smooth: float = 1e-3) -> None:
            Initializes the DiceLoss instance.
        forward(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            Computes the Dice loss.

    Returns:
        torch.Tensor: The computed Dice loss.

    Examples:
        >>> loss_fn = DiceLoss()
        >>> input = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]])
        >>> target = torch.tensor([[[[0, 1], [1, 0]], [[1, 0], [0, 1]]]])
        >>> loss = loss_fn(input, target)
    """

    def __init__(self, smooth: float = 1e-3) -> None:
        super().__init__(size_average=None, reduce=None, reduction="mean")
        self.smooth = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the Dice loss.

        Args:
            input (torch.Tensor): Prediction array with shape (N, C, H, W) where N is the batch size,
                C is the number of classes, H is the height, and W is the width.
            target (torch.Tensor): Ground truth array with shape (N, C, H, W) where N is the batch size,
                C is the number of classes, H is the height, and W is the width.

        Returns:
            torch.Tensor: The computed Dice loss.

        Raises:
            AssertionError: If the input and target tensors have different shapes.

        Examples:
            >>> loss_fn = DiceLoss()
            >>> input = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]])
            >>> target = torch.tensor([[[[0, 1], [1, 0]], [[1, 0], [0, 1]]]])
            >>> loss = loss_fn(input, target)
        """
        assert (
            input.shape == target.shape
        ), "Input and target tensors must have the same shape"

        input = input.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        inse = torch.sum(input * target, (0, 1, 2))
        l = torch.sum(input, (0, 1, 2))
        r = torch.sum(target, (0, 1, 2))
        loss = 1.0 - (2.0 * inse + self.smooth) / (l + r + self.smooth)
        loss = torch.sum(loss)

        return loss


class MSELossMaps(_Loss):
    """Mean Squared Error Loss for Combined Horizontal and Vertical Maps.

    This class represents the mean squared error loss function for combined horizontal and vertical maps
    in segmentation tasks.

    Args:
        None

    Methods:
        __init__() -> None:
            Initializes the MSELossMaps instance.
        forward(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            Computes the mean squared error loss.

    Returns:
        torch.Tensor: The computed mean squared error loss per pixel.

    Examples:
        >>> loss_fn = MSELossMaps()
        >>> input = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]])
        >>> target = torch.tensor([[[[0.2, 0.3], [0.4, 0.5]], [[0.6, 0.7], [0.8, 0.9]]]])
        >>> loss = loss_fn(input, target)
    """

    def __init__(self) -> None:
        super().__init__(size_average=None, reduce=None, reduction="mean")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean squared error loss.

        Args:
            input (torch.Tensor): Prediction of combined horizontal and vertical maps
                with shape (N, 2, H, W), where channel 0 is vertical and channel 1 is horizontal.
            target (torch.Tensor): Ground truth of combined horizontal and vertical maps
                with shape (N, 2, H, W), where channel 0 is vertical and channel 1 is horizontal.

        Returns:
            torch.Tensor: The computed mean squared error loss per pixel with shape (N, 2, H, W).

        Examples:
            >>> loss_fn = MSELossMaps()
            >>> input = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]])
            >>> target = torch.tensor([[[[0.2, 0.3], [0.4, 0.5]], [[0.6, 0.7], [0.8, 0.9]]]])
            >>> loss = loss_fn(input, target)
        """
        loss = (input - target) ** 2
        loss = loss.mean()

        return loss


class MSGELossMaps(_Loss):
    """Mean Squared Gradient Error Loss for Combined Horizontal and Vertical Maps.

    This class represents the mean squared gradient error loss function for combined horizontal and vertical maps
    in segmentation tasks.

    Args:
        None

    Methods:
        get_sobel_kernel(size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
            Get the Sobel kernel with a given size.
        get_gradient_hv(hv: torch.Tensor, device: str) -> torch.Tensor:
            Calculate the gradient of the horizontal and vertical prediction map.
        forward(input: torch.Tensor, target: torch.Tensor, focus: torch.Tensor, device: str) -> torch.Tensor:
            Computes the mean squared gradient error loss.

    Returns:
        torch.Tensor: The computed mean squared gradient error loss.

    Examples:
        >>> loss_fn = MSGELossMaps()
        >>> input = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]])
        >>> target = torch.tensor([[[[0.2, 0.3], [0.4, 0.5]], [[0.6, 0.7], [0.8, 0.9]]]])
        >>> focus = torch.tensor([[[[1, 1], [1, 1]], [[1, 1], [1, 1]]]])
        >>> loss = loss_fn(input, target, focus, "cuda")
    """

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
        """MSGE (Gradient of MSE) loss

        Args:
            input (torch.Tensor): Input with shape (B, C, H, W)
            target (torch.Tensor): Target with shape (B, C, H, W)
            focus (torch.Tensor): Focus, type of masking (B, C, W, W)
            device (str): CUDA device to work with.

        Returns:
            torch.Tensor: MSGE loss
        """
        input = input.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        focus = focus.permute(0, 2, 3, 1)
        focus = focus[..., 1]

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
    """Focal Tversky Loss Function for Multiple Classes.

    This class implements the Focal Tversky Loss function for multiple classes in PyTorch.
    The Focal Tversky Loss is a modification of the Tversky Loss that introduces a focal parameter
    to emphasize hard examples during training.

    Reference:
    Abraham, N., & Khan, N. M. (2019).
    A Novel Focal Tversky Loss Function With Improved Attention U-Net for Lesion Segmentation.
    In International Symposium on Biomedical Imaging. https://doi.org/10.1109/isbi.2019.8759329

    Args:
        alpha_t (float, optional): Alpha parameter for Tversky loss (multiplied with false-negatives). Defaults to 0.7.
        beta_t (float, optional): Beta parameter for Tversky loss (multiplied with false-positives). Defaults to 0.3.
        gamma_f (float, optional): Gamma Focal parameter. Defaults to 4/3.
        smooth (float, optional): Smoothing factor. Defaults to 1e-6.

    Attributes:
        alpha_t (float): Alpha parameter for Tversky loss (multiplied with false-negatives).
        beta_t (float): Beta parameter for Tversky loss (multiplied with false-positives).
        gamma_f (float): Gamma Focal parameter.
        smooth (float): Smoothing factor.
        num_classes (int): Number of classes.

    Methods:
        forward(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            Calculates the Focal Tversky Loss.

    Raises:
        ValueError: If there is a shape mismatch between predictions and targets.

    Returns:
        torch.Tensor: The calculated Focal Tversky Loss.

    Examples:
        >>> loss_fn = FocalTverskyLoss()
        >>> input = torch.tensor([[[[0.1, 0.9], [0.3, 0.7]], [[0.6, 0.4], [0.8, 0.2]]]])
        >>> target = torch.tensor([[[[1, 0], [0, 1]], [[0, 1], [1, 0]]]])
        >>> loss = loss_fn(input, target)
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
            input (torch.Tensor): Predictions, logits (without Softmax). Shape: (B, C, H, W)
            target (torch.Tensor): Targets, either flattened (Shape: (C, H, W) or as one-hot encoded (Shape: (batch-size, C, H, W)).

        Raises:
            ValueError: Error if there is a shape missmatch

        Returns:
            torch.Tensor: FocalTverskyLoss (weighted)
        """
        input = input.permute(0, 2, 3, 1)
        if input.shape[-1] != self.num_classes:
            raise ValueError(
                "Predictions must be a logit tensor with the last dimension shape beeing equal to the number of classes"
            )
        if len(target.shape) != len(input.shape):
            # convert the targets to onehot
            target = F.one_hot(target, num_classes=self.num_classes)

        # flatten
        target = target.permute(0, 2, 3, 1)
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

    This class implements the Multiclass Focal Tversky Loss function for multiple classes in PyTorch.
    The Focal Tversky Loss is a modification of the Tversky Loss that introduces a focal parameter
    to emphasize hard examples during training.

    Reference:
    Abraham, N., & Khan, N. M. (2019).
    A Novel Focal Tversky Loss Function With Improved Attention U-Net for Lesion Segmentation.
    In International Symposium on Biomedical Imaging. https://doi.org/10.1109/isbi.2019.8759329

    Args:
        alpha_t (float, optional): Alpha parameter for Tversky loss (multiplied with false-negatives). Defaults to 0.7.
        beta_t (float, optional): Beta parameter for Tversky loss (multiplied with false-positives). Defaults to 0.3.
        gamma_f (float, optional): Gamma Focal parameter. Defaults to 4/3.
        smooth (float, optional): Smoothing factor. Defaults to 0.000001.
        num_classes (int, optional): Number of output classes. For binary segmentation, prefer FocalTverskyLoss (speed optimized). Defaults to 2.
        class_weights (List[int], optional): Weights for each class. If not provided, equal weight. Length must be equal to num_classes. Defaults to None.

    Attributes:
        alpha_t (float): Alpha parameter for Tversky loss (multiplied with false-negatives).
        beta_t (float): Beta parameter for Tversky loss (multiplied with false-positives).
        gamma_f (float): Gamma Focal parameter.
        smooth (float): Smoothing factor.
        num_classes (int): Number of output classes.
        class_weights (torch.Tensor): Weights for each class.

    Methods:
        forward(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            Calculates the Multiclass Focal Tversky Loss.

    Raises:
        ValueError: If there is a shape mismatch between predictions and targets.

    Returns:
        torch.Tensor: The calculated Multiclass Focal Tversky Loss.

    Examples:
        >>> loss_fn = MCFocalTverskyLoss()
        >>> input = torch.tensor([[[[0.1, 0.9], [0.3, 0.7]], [[0.6, 0.4], [0.8, 0.2]]]])
        >>> target = torch.tensor([[[[1, 0], [0, 1]], [[0, 1], [1, 0]]]])
        >>> loss = loss_fn(input, target)
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
            input (torch.Tensor): Predictions, logits (without Softmax). Shape: (B, num_classes, H, W)
            target (torch.Tensor): Targets, either flattened (Shape: (B, H, W) or as one-hot encoded (Shape: (B, num_classes, H, W)).

        Raises:
            ValueError: Error if there is a shape missmatch

        Returns:
            torch.Tensor: FocalTverskyLoss (weighted)
        """
        input = input.permute(0, 2, 3, 1)
        if input.shape[-1] != self.num_classes:
            raise ValueError(
                "Predictions must be a logit tensor with the last dimension shape beeing equal to the number of classes"
            )
        if len(target.shape) != len(input.shape):
            # convert the targets to onehot
            target = F.one_hot(target, num_classes=self.num_classes)

        target = target.permute(0, 2, 3, 1)
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


class WeightedBaseLoss(nn.Module):
    """Base class for weighted cross entropy based losses.

    Adapted/Copied from: https://github.com/okunator/cellseg_models.pytorch (10.5281/zenodo.7064617)

    This class provides a base implementation for weighted cross entropy based losses.
    It enables weighting for object instance edges and classes.

    Args:
        apply_sd (bool, optional): If True, Spectral decoupling regularization will be applied to the loss matrix. Defaults to False.
        apply_ls (bool, optional): If True, Label smoothing will be applied to the target. Defaults to False.
        apply_svls (bool, optional): If True, spatially varying label smoothing will be applied to the target. Defaults to False.
        apply_mask (bool, optional): If True, a mask will be applied to the loss matrix. Defaults to False.
        class_weights (torch.Tensor, optional): Class weights. A tensor of shape (C, ). Defaults to None.
        edge_weight (float, optional): Weight for the object instance border pixels. Defaults to None.

    Attributes:
        apply_sd (bool): If True, Spectral decoupling regularization will be applied to the loss matrix.
        apply_ls (bool): If True, Label smoothing will be applied to the target.
        apply_svls (bool): If True, spatially varying label smoothing will be applied to the target.
        apply_mask (bool): If True, a mask will be applied to the loss matrix.
        class_weights (torch.Tensor): Class weights.
        edge_weight (float): Weight for the object instance border pixels.

    Methods:
        apply_spectral_decouple(loss_matrix: torch.Tensor, yhat: torch.Tensor, lam: float = 0.01) -> torch.Tensor:
            Apply spectral decoupling L2 norm after the loss.
        apply_ls_to_target(target: torch.Tensor, num_classes: int, label_smoothing: float = 0.1) -> torch.Tensor:
            Apply label smoothing to the target.
        apply_svls_to_target(target: torch.Tensor, num_classes: int, kernel_size: int = 5, sigma: int = 3, **kwargs) -> torch.Tensor:
            Apply spatially varying label smoothing to the target map.
        apply_class_weights(loss_matrix: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            Multiply pixelwise loss matrix by the class weights.
        apply_edge_weights(loss_matrix: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
            Apply weights to the object boundaries.
        apply_mask_weight(loss_matrix: torch.Tensor, mask: torch.Tensor, norm: bool = True) -> torch.Tensor:
            Apply a mask to the loss matrix.

    Examples:
        >>> loss_fn = WeightedBaseLoss(apply_sd=True, apply_ls=True, apply_svls=True, apply_mask=True, class_weights=torch.tensor([0.5, 1.0]), edge_weight=0.2)
        >>> loss_matrix = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        >>> yhat = torch.tensor([[0.5, 0.5], [0.6, 0.4]])
        >>> loss_matrix_sd = loss_fn.apply_spectral_decouple(loss_matrix, yhat)
        >>> target = torch.tensor([[1, 0], [0, 1]])
        >>> loss_matrix_ls = loss_fn.apply_ls_to_target(target, num_classes=2)
        >>> loss_matrix_svls = loss_fn.apply_svls_to_target(target, num_classes=2)
        >>> loss_matrix_weighted = loss_fn.apply_class_weights(loss_matrix, target)
        >>> weight_map = torch.tensor([[0, 1], [1, 0]])
        >>> loss_matrix_edge_weighted = loss_fn.apply_edge_weights(loss_matrix, weight_map)
        >>> mask = torch.tensor([[1, 1], [1, 1]])
        >>> loss_matrix_masked = loss_fn.apply_mask_weight(loss_matrix, mask)
    """

    def __init__(
        self,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
        apply_mask: bool = False,
        class_weights: torch.Tensor = None,
        edge_weight: float = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.apply_sd = apply_sd
        self.apply_ls = apply_ls
        self.apply_svls = apply_svls
        self.apply_mask = apply_mask
        self.class_weights = class_weights
        self.edge_weight = edge_weight

    def apply_spectral_decouple(
        self, loss_matrix: torch.Tensor, yhat: torch.Tensor, lam: float = 0.01
    ) -> torch.Tensor:
        """Apply spectral decoupling L2 norm after the loss.

        https://arxiv.org/abs/2011.09468

        Args:
            loss_matrix (torch.Tensor): Pixelwise losses. A tensor of shape (B, H, W).
            yhat (torch.Tensor): The pixel predictions of the model. Shape (B, C, H, W).
            lam (float, optional): Lambda constant.. Defaults to 0.01.

        Returns:
            torch.Tensor: SD-regularized loss matrix. Same shape as input.
        """
        return loss_matrix + (lam / 2) * (yhat**2).mean(axis=1)

    def apply_ls_to_target(
        self,
        target: torch.Tensor,
        num_classes: int,
        label_smoothing: float = 0.1,
    ) -> torch.Tensor:
        """Apply label smoothing to the target.

        Args:
            target (torch.Tensor): Number of classes in the data.
            num_classes (int): The target one hot tensor. Shape (B, C, H, W)
            label_smoothing (float, optional):  The smoothing coeff alpha. Defaults to 0.1.

        Returns:
            torch.Tensor: Label smoothed target. Same shape as input.
        """
        return target * (1 - label_smoothing) + label_smoothing / num_classes

    def apply_svls_to_target(
        self,
        target: torch.Tensor,
        num_classes: int,
        kernel_size: int = 5,
        sigma: int = 3,
        **kwargs,
    ) -> torch.Tensor:
        """Apply spatially varying label smoothihng to target map.

        https://arxiv.org/abs/2104.05788

        Args:
            target (torch.Tensor): The target one hot tensor. Shape (B, C, H, W).
            num_classes (int):  Number of classes in the data.
            kernel_size (int, optional): Size of a square kernel.. Defaults to 5.
            sigma (int, optional): The std of the gaussian. Defaults to 3.

        Returns:
            torch.Tensor: Label smoothed target. Same shape as input.
        """
        my, mx = kernel_size // 2, kernel_size // 2
        gaussian_kernel = gaussian_kernel2d(
            kernel_size, sigma, num_classes, device=target.device
        )
        neighborsum = (1 - gaussian_kernel[..., my, mx]) + 1e-16
        gaussian_kernel = gaussian_kernel.clone()
        gaussian_kernel[..., my, mx] = neighborsum
        svls_kernel = gaussian_kernel / neighborsum[0]

        return filter2D(target.float(), svls_kernel) / svls_kernel[0].sum()

    def apply_class_weights(
        self, loss_matrix: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Multiply pixelwise loss matrix by the class weights.

        Note: No normalization

        Args:
            loss_matrix (torch.Tensor): Pixelwise losses. A tensor of shape (B, H, W).
            target (torch.Tensor): The target mask. Shape (B, H, W).

        Returns:
            torch.Tensor: The loss matrix scaled with the weight matrix. Shape (B, H, W).
        """
        weight_mat = self.class_weights[target.long()].to(target.device)  # to (B, H, W)
        loss = loss_matrix * weight_mat

        return loss

    def apply_edge_weights(
        self, loss_matrix: torch.Tensor, weight_map: torch.Tensor
    ) -> torch.Tensor:
        """Apply weights to the object boundaries.

        Basically just computes `edge_weight`**`weight_map`.

        Args:
            loss_matrix (torch.Tensor): Pixelwise losses. A tensor of shape (B, H, W).
            weight_map (torch.Tensor): Map that points to the pixels that will be weighted. Shape (B, H, W).

        Returns:
            torch.Tensor: The loss matrix scaled with the nuclear boundary weights. Shape (B, H, W).
        """
        return loss_matrix * self.edge_weight**weight_map

    def apply_mask_weight(
        self, loss_matrix: torch.Tensor, mask: torch.Tensor, norm: bool = True
    ) -> torch.Tensor:
        """Apply a mask to the loss matrix.

        Args:
            loss_matrix (torch.Tensor): Pixelwise losses. A tensor of shape (B, H, W).
            mask (torch.Tensor): The mask. Shape (B, H, W).
            norm (bool, optional): If True, the loss matrix will be normalized by the mean of the mask. Defaults to True.

        Returns:
            torch.Tensor: The loss matrix scaled with the mask. Shape (B, H, W).
        """
        loss_matrix *= mask
        if norm:
            norm_mask = torch.mean(mask.float()) + 1e-7
            loss_matrix /= norm_mask

        return loss_matrix

    def extra_repr(self) -> str:
        """Add info to print."""
        s = "apply_sd={apply_sd}, apply_ls={apply_ls}, apply_svls={apply_svls}, apply_mask={apply_mask}, class_weights={class_weights}, edge_weight={edge_weight}"  # noqa
        return s.format(**self.__dict__)


class MAEWeighted(WeightedBaseLoss):
    """Compute the Mean Absolute Error (MAE) loss with optional regularization.

    This class calculates the MAE loss between the predicted map and the ground truth annotations.
    It supports optional regularization techniques such as spectral decoupling and edge weights.

    Reference:
    Stardist: https://arxiv.org/pdf/1806.03535.pdf
    Adapted/Copied from: https://github.com/okunator/cellseg_models.pytorch (10.5281/zenodo.7064617)

    Args:
        alpha (float, optional): Weight regularizer between 0 and 1. Defaults to 1e-4.
        apply_sd (bool, optional): If True, apply spectral decoupling regularization to the loss matrix. Defaults to False.
        apply_mask (bool, optional): If True, apply a mask to the loss matrix. Defaults to False.
        edge_weight (float, optional): Weight added to object borders. Defaults to None.

    Attributes:
        alpha (float): Weight regularizer between 0 and 1.
        eps (float): Small epsilon value for numerical stability.

    Methods:
        forward(input: torch.Tensor, target: torch.Tensor, target_weight: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
            Compute the masked MAE loss.

    Raises:
        ValueError: If the shapes of the predictions and targets do not match.

    Returns:
        torch.Tensor: Computed MAE loss (scalar).

    Examples:
        >>> loss_fn = MAEWeighted()
        >>> input = torch.tensor([[[[0.1, 0.9], [0.3, 0.7]], [[0.6, 0.4], [0.8, 0.2]]]])
        >>> target = torch.tensor([[[1, 0], [0, 1]]])
        >>> loss = loss_fn(input, target)
    """

    def __init__(
        self,
        alpha: float = 1e-4,
        apply_sd: bool = False,
        apply_mask: bool = False,
        edge_weight: float = None,
        **kwargs,
    ) -> None:
        super().__init__(apply_sd, False, False, apply_mask, False, edge_weight)
        self.alpha = alpha
        self.eps = 1e-7

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the masked MAE loss.

        Args:
            input (torch.Tensor): The prediction map. Shape (B, C, H, W).
            target (torch.Tensor): The ground truth annotations. Shape (B, H, W).
            target_weight (torch.Tensor, optional): The edge weight map. Shape (B, H, W). Defaults to None.
            mask (torch.Tensor, optional): The mask map. Shape (B, H, W). Defaults to None.

        Raises:
            ValueError: Pred and target shapes must match.

        Returns:
            torch.Tensor: Computed MAE loss (scalar).
        """
        yhat = input
        n_classes = yhat.shape[1]
        if target.size() != yhat.size():
            target = target.unsqueeze(1).repeat_interleave(n_classes, dim=1)

        if not yhat.shape == target.shape:
            raise ValueError(
                f"Pred and target shapes must match. Got: {yhat.shape}, {target.shape}"
            )

        # compute the MAE loss with alpha as weight
        mae_loss = torch.mean(torch.abs(target - yhat), axis=1)  # (B, H, W)

        if self.apply_mask and mask is not None:
            mae_loss = self.apply_mask_weight(mae_loss, mask, norm=True)  # (B, H, W)

            # add the background regularization
            if self.alpha > 0:
                reg = torch.mean(((1 - mask).unsqueeze(1)) * torch.abs(yhat), axis=1)
                mae_loss += self.alpha * reg

        if self.apply_sd:
            mae_loss = self.apply_spectral_decouple(mae_loss, yhat)

        if self.edge_weight is not None:
            mae_loss = self.apply_edge_weights(mae_loss, target_weight)

        return mae_loss.mean()


class MSEWeighted(WeightedBaseLoss):
    """MSE-loss.

    Args:
        apply_sd (bool, optional): If True, Spectral decoupling regularization will be applied to the
            loss matrix. Defaults to False.
        apply_ls (bool, optional): If True, Label smoothing will be applied to the target. Defaults to False.
        apply_svls (bool, optional): If True, spatially varying label smoothing will be applied to the target. Defaults to False.
        apply_mask (bool, optional): If True, a mask will be applied to the loss matrix. Mask shape: (B, H, W). Defaults to False.
        edge_weight (float, optional): Weight that is added to object borders. Defaults to None.
        class_weights (torch.Tensor, optional): Class weights. A tensor of shape (n_classes,). Defaults to None.

    Attributes:
        None

    Methods:
        tensor_one_hot(type_map: torch.Tensor, n_classes: int) -> torch.Tensor:
            Convert a segmentation mask into one-hot-format.
        forward(input: torch.Tensor, target: torch.Tensor, target_weight: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
            Compute the MSE-loss.

    Raises:
        TypeError: If the input `type_map` does not have dtype torch.int64.

    Returns:
        torch.Tensor: Computed MSE loss (scalar).

    Examples:
        >>> loss_fn = MSEWeighted()
        >>> input = torch.tensor([[[[0.1, 0.9], [0.3, 0.7]], [[0.6, 0.4], [0.8, 0.2]]]])
        >>> target = torch.tensor([[[1, 0], [0, 1]]])
        >>> loss = loss_fn(input, target)
    """

    def __init__(
        self,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
        apply_mask: bool = False,
        edge_weight: float = None,
        class_weights: torch.Tensor = None,
        **kwargs,
    ) -> None:
        super().__init__(
            apply_sd, apply_ls, apply_svls, apply_mask, class_weights, edge_weight
        )

    @staticmethod
    def tensor_one_hot(type_map: torch.Tensor, n_classes: int) -> torch.Tensor:
        """Convert a segmentation mask into one-hot-format.

        I.e. Takes in a segmentation mask of shape (B, H, W) and reshapes it
        into a tensor of shape (B, C, H, W).

        Args:
            type_map (torch.Tensor):  Multi-label Segmentation mask. Shape (B, H, W).
            n_classes (int): Number of classes. (Zero-class included.)

        Raises:
            TypeError: Input `type_map` should have dtype: torch.int64.

        Returns:
            torch.Tensor: A one hot tensor. Shape: (B, C, H, W). Dtype: torch.FloatTensor.
        """
        if not type_map.dtype == torch.int64:
            raise TypeError(
                f"""
                Input `type_map` should have dtype: torch.int64. Got: {type_map.dtype}."""
            )

        one_hot = torch.zeros(
            type_map.shape[0],
            n_classes,
            *type_map.shape[1:],
            device=type_map.device,
            dtype=type_map.dtype,
        )

        return one_hot.scatter_(dim=1, index=type_map.unsqueeze(1), value=1.0) + 1e-7

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the MSE-loss.

        Args:
            input (torch.Tensor): The prediction map. Shape (B, C, H, W, C).
            target (torch.Tensor): The ground truth annotations. Shape (B, H, W).
            target_weight (torch.Tensor, optional):  The edge weight map. Shape (B, H, W). Defaults to None.
            mask (torch.Tensor, optional): The mask map. Shape (B, H, W). Defaults to None.

        Returns:
            torch.Tensor: Computed MSE loss (scalar).
        """
        yhat = input
        target_one_hot = target
        num_classes = yhat.shape[1]

        if target.size() != yhat.size():
            if target.dtype == torch.float32:
                target_one_hot = target.unsqueeze(1)
            else:
                target_one_hot = MSEWeighted.tensor_one_hot(target, num_classes)

        if self.apply_svls:
            target_one_hot = self.apply_svls_to_target(
                target_one_hot, num_classes, **kwargs
            )

        if self.apply_ls:
            target_one_hot = self.apply_ls_to_target(
                target_one_hot, num_classes, **kwargs
            )

        mse = F.mse_loss(yhat, target_one_hot, reduction="none")  # (B, C, H, W)
        mse = torch.mean(mse, dim=1)  # to (B, H, W)

        if self.apply_mask and mask is not None:
            mse = self.apply_mask_weight(mse, mask, norm=False)  # (B, H, W)

        if self.apply_sd:
            mse = self.apply_spectral_decouple(mse, yhat)

        if self.class_weights is not None:
            mse = self.apply_class_weights(mse, target)

        if self.edge_weight is not None:
            mse = self.apply_edge_weights(mse, target_weight)

        return torch.mean(mse)


class BCEWeighted(WeightedBaseLoss):
    """Binary Cross Entropy Loss with Weighting and Other Tricks.

    Args:
        apply_sd (bool, optional): If True, Spectral decoupling regularization will be applied to the
            loss matrix. Defaults to False.
        apply_ls (bool, optional): If True, Label smoothing will be applied to the target. Defaults to False.
        apply_svls (bool, optional): If True, spatially varying label smoothing will be applied to the target. Defaults to False.
        apply_mask (bool, optional): If True, a mask will be applied to the loss matrix. Mask shape: (B, H, W). Defaults to False.
        edge_weight (float, optional): Weight that is added to object borders. Defaults to None.
        class_weights (torch.Tensor, optional): Class weights. A tensor of shape (n_classes,). Defaults to None.

    Attributes:
        eps (float): A small value used for numerical stability.

    Methods:
        forward(input: torch.Tensor, target: torch.Tensor, target_weight: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
            Compute the binary cross entropy loss.

    Raises:
        None

    Returns:
        torch.Tensor: Computed binary cross entropy loss (scalar).

    Examples:
        >>> loss_fn = BCEWeighted()
        >>> input = torch.tensor([[[[0.1, 0.9], [0.3, 0.7]], [[0.6, 0.4], [0.8, 0.2]]]])
        >>> target = torch.tensor([[[1, 0], [0, 1]]])
        >>> loss = loss_fn(input, target)
    """

    def __init__(
        self,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
        apply_mask: bool = False,
        edge_weight: float = None,
        class_weights: torch.Tensor = None,
        **kwargs,
    ) -> None:
        super().__init__(
            apply_sd, apply_ls, apply_svls, apply_mask, class_weights, edge_weight
        )
        self.eps = 1e-8

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the binary cross entropy loss.

        Args:
            input (torch.Tensor): The prediction map. Shape (B, C, H, W).
            target (torch.Tensor): The ground truth annotations. Shape (B, H, W).
            target_weight (torch.Tensor, optional): The edge weight map. Shape (B, H, W). Defaults to None.
            mask (torch.Tensor, optional): The mask map. Shape (B, H, W). Defaults to None.

        Returns:
            torch.Tensor: Computed binary cross entropy loss (scalar).
        """
        yhat = input
        num_classes = yhat.shape[1]
        yhat = torch.clip(yhat, self.eps, 1.0 - self.eps)

        if target.size() != yhat.size():
            target = target.unsqueeze(1).repeat_interleave(num_classes, dim=1)

        if self.apply_svls:
            target = self.apply_svls_to_target(target, num_classes, **kwargs)

        if self.apply_ls:
            target = self.apply_ls_to_target(target, num_classes, **kwargs)

        bce = F.binary_cross_entropy_with_logits(
            yhat.float(), target.float(), reduction="none"
        )  # (B, C, H, W)
        bce = torch.mean(bce, dim=1)  # (B, H, W)

        if self.apply_mask and mask is not None:
            bce = self.apply_mask_weight(bce, mask, norm=False)  # (B, H, W)

        if self.apply_sd:
            bce = self.apply_spectral_decouple(bce, yhat)

        if self.class_weights is not None:
            bce = self.apply_class_weights(bce, target)

        if self.edge_weight is not None:
            bce = self.apply_edge_weights(bce, target_weight)

        return torch.mean(bce)


class CEWeighted(WeightedBaseLoss):
    """Cross-Entropy Loss with Weighting.

    This class represents the cross-entropy loss function with weighting for classification tasks.

    Args:
        apply_sd (bool, optional): If True, Spectral decoupling regularization will be applied to the
            loss matrix. Defaults to False.
        apply_ls (bool, optional): If True, Label smoothing will be applied to the target. Defaults to False.
        apply_svls (bool, optional): If True, spatially varying label smoothing will be applied to the target. Defaults to False.
        apply_mask (bool, optional): If True, a mask will be applied to the loss matrix. Mask shape: (B, H, W). Defaults to False.
        edge_weight (float, optional): Weight that is added to object borders. Defaults to None.
        class_weights (torch.Tensor, optional): Class weights. A tensor of shape (n_classes,). Defaults to None.

    Attributes:
        eps (float): A small value used for numerical stability.

    Methods:
        forward(input: torch.Tensor, target: torch.Tensor, target_weight: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
            Compute the cross-entropy loss.

    Raises:
        None

    Returns:
        torch.Tensor: Computed cross-entropy loss (scalar).

    Examples:
        >>> loss_fn = CEWeighted()
        >>> input = torch.tensor([[[[0.1, 0.9], [0.3, 0.7]], [[0.6, 0.4], [0.8, 0.2]]]])
        >>> target = torch.tensor([[[1, 0], [0, 1]]])
        >>> loss = loss_fn(input, target)
    """

    def __init__(
        self,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
        apply_mask: bool = False,
        edge_weight: float = None,
        class_weights: torch.Tensor = None,
        **kwargs,
    ) -> None:
        super().__init__(
            apply_sd, apply_ls, apply_svls, apply_mask, class_weights, edge_weight
        )
        self.eps = 1e-8

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the cross-entropy loss.

        Args:
            input (torch.Tensor): The prediction map. Shape (B, C, H, W).
            target (torch.Tensor): The ground truth annotations. Shape (B, H, W).
            target_weight (torch.Tensor, optional): The edge weight map. Shape (B, H, W). Defaults to None.
            mask (torch.Tensor, optional): The mask map. Shape (B, H, W). Defaults to None.

        Returns:
            torch.Tensor: Computed cross-entropy loss (scalar).
        """
        yhat = input  # TODO: remove doubled Softmax -> this function needs logits instead of softmax output
        input_soft = F.softmax(yhat, dim=1) + self.eps  # (B, C, H, W)
        num_classes = yhat.shape[1]
        if len(target.shape) != len(yhat.shape) and target.shape[1] != num_classes:
            target_one_hot = MSEWeighted.tensor_one_hot(
                target, num_classes
            )  # (B, C, H, W)
        else:
            target_one_hot = target
            target = torch.argmax(target, dim=1)
        assert target_one_hot.shape == yhat.shape

        if self.apply_svls:
            target_one_hot = self.apply_svls_to_target(
                target_one_hot, num_classes, **kwargs
            )

        if self.apply_ls:
            target_one_hot = self.apply_ls_to_target(
                target_one_hot, num_classes, **kwargs
            )

        loss = -torch.sum(target_one_hot * torch.log(input_soft), dim=1)  # (B, H, W)

        if self.apply_mask and mask is not None:
            loss = self.apply_mask_weight(loss, mask, norm=False)  # (B, H, W)

        if self.apply_sd:
            loss = self.apply_spectral_decouple(loss, yhat)

        if self.class_weights is not None:
            loss = self.apply_class_weights(loss, target)

        if self.edge_weight is not None:
            loss = self.apply_edge_weights(loss, target_weight)

        return loss.mean()


class L1LossWeighted(nn.Module):
    """Weighted L1 Loss.

    This class represents the weighted L1 loss function for regression tasks.

    Methods:
        forward(input: torch.Tensor, target: torch.Tensor, target_weight: torch.Tensor = None) -> torch.Tensor:
            Computes the weighted L1 loss.

    Args:
        input (torch.Tensor): The predicted values.
        target (torch.Tensor): The ground truth values.
        target_weight (torch.Tensor, optional): The weight for each target value. Defaults to None.

    Returns:
        torch.Tensor: The computed weighted L1 loss.

    Examples:
        >>> loss_fn = L1LossWeighted()
        >>> input = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
        >>> target = torch.tensor([[[0.2, 0.3], [0.4, 0.5]], [[0.6, 0.7], [0.8, 0.9]]])
        >>> loss = loss_fn(input, target)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Computes the weighted L1 loss.

        Args:
            input (torch.Tensor): The predicted values.
            target (torch.Tensor): The ground truth values.
            target_weight (torch.Tensor, optional): The weight for each target value. Defaults to None.

        Returns:
            torch.Tensor: The computed weighted L1 loss.
        """
        l1loss = F.l1_loss(input, target, size_average=True, reduce=False)
        l1loss = torch.mean(l1loss, dim=1)
        if target_weight is not None:
            l1loss = torch.mean(target_weight * l1loss)
        else:
            l1loss = torch.mean(l1loss)
        return l1loss


def retrieve_loss_fn(loss_name: dict, **kwargs) -> _Loss:
    """Return the loss function with the given name defined in the LOSS_DICT and initialize it with kwargs.

    Args:
        loss_name (dict): Name of the loss function.

    Returns:
        _Loss: Loss function.

    Raises:
        KeyError: If the loss_name is not found in the LOSS_DICT.

    Examples:
        >>> loss_fn = retrieve_loss_fn("L1LossWeighted", apply_sd=True, apply_ls=False)
        >>> print(loss_fn)
        L1LossWeighted(apply_sd=True, apply_ls=False)
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
    "CrossEntropyLoss": nn.CrossEntropyLoss,  # input logits, targets
    "L1Loss": nn.L1Loss,
    "MSELoss": nn.MSELoss,
    "CTCLoss": nn.CTCLoss,  # probability
    "NLLLoss": nn.NLLLoss,  # log-probabilities of each class
    "PoissonNLLLoss": nn.PoissonNLLLoss,
    "GaussianNLLLoss": nn.GaussianNLLLoss,
    "KLDivLoss": nn.KLDivLoss,  # argument input in log-space
    "BCELoss": nn.BCELoss,  # probabilities
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,  # logits
    "MarginRankingLoss": nn.MarginRankingLoss,
    "HingeEmbeddingLoss": nn.HingeEmbeddingLoss,
    "MultiLabelMarginLoss": nn.MultiLabelMarginLoss,
    "HuberLoss": nn.HuberLoss,
    "SmoothL1Loss": nn.SmoothL1Loss,
    "SoftMarginLoss": nn.SoftMarginLoss,  # logits
    "MultiLabelSoftMarginLoss": nn.MultiLabelSoftMarginLoss,
    "CosineEmbeddingLoss": nn.CosineEmbeddingLoss,
    "MultiMarginLoss": nn.MultiMarginLoss,
    "TripletMarginLoss": nn.TripletMarginLoss,
    "TripletMarginWithDistanceLoss": nn.TripletMarginWithDistanceLoss,
    "MAEWeighted": MAEWeighted,
    "MSEWeighted": MSEWeighted,
    "BCEWeighted": BCEWeighted,  # logits
    "CEWeighted": CEWeighted,  # logits
    "L1LossWeighted": L1LossWeighted,
}
