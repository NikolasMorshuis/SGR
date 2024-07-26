import torch
from torch import nn
from typing import Callable

__all__ = ["SenseModel"]

class SenseModel(nn.Module):
    """
    A module that computes forward and adjoint SENSE operations.

    The forward operation converts a complex image -> multi-coil kspace.
    The adjoint operation converts multi-coil kspace -> a complex image.

    This module also supports multiple sensitivity maps. This is useful if
    you would like to generate images from multiple estimated sensitivity maps.
    This module also works with single coil inputs as long as the #coils dimension
    is set to 1.

    Attributes:
        maps (torch.Tensor): Sensitivity maps. Shape ``(B, H, W, #coils, #maps, [2])``.
        weights (torch.Tensor, optional): Undersampling masks (if applicable).
            Shape ``(B, H, W)`` or ``(B, H, W, #coils, #coils)``.
    """

    def __init__(self, maps: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            maps (torch.Tensor): Sensitivity maps.
            weights (torch.Tensor): Undersampling masks.
                If ``None``, it is assumed that inputs are fully-sampled.
        """
        super().__init__()

        self.maps = maps  # [B, H, W, #coils]
        if weights is None:
            self.weights = 1.0
        else:
            self.weights = weights

    def _adjoint_op(self, kspace):
        """
        Args:
            kspace: Shape (B,H,W,#coils,[2])
        Returns:
            image: Shape (B,H,W,#maps,[2])
        """
        image = torch.fft.ifftshift(kspace, dim=(1, 2))
        image = torch.fft.ifft2(image, dim=(1, 2), norm="ortho")
        image = torch.fft.fftshift(image, dim=(1, 2))
        image = image * torch.conj(self.maps)
        return image.sum(-1)

    def _forward_op(self, image):
        """
        Args:
            image: Shape (B,H,W,#maps,[2])
        Returns:
            kspace: Shape (B,H,W,#coils,[2])
        """

        kspace = image.unsqueeze(-1) * self.maps
        kspace = torch.fft.fftshift(kspace, dim=(1, 2))
        kspace = torch.fft.fft2(kspace, dim=(1, 2), norm="ortho")
        kspace = torch.fft.ifftshift(kspace, dim=(1, 2))
        kspace = self.weights * kspace
        return kspace

    def forward_backward(self, image):
        """
        equivalent to A* A
        """
        kspace = self._forward_op(image)
        image = self._adjoint_op(kspace)
        return image

    def CG(self, kspace_us, x, mask, max_iter=10):
        """
        b is the undersampled image(!) and x is the initial guess
        """
        is_complex=True
        if not torch.is_complex(x):
            # we have (c, h, w)
            is_complex=False
            x = torch.complex(x[:, 0], x[:, 1])

        b_kspace = mask * kspace_us + ~mask * self._forward_op(x)
        b = self._adjoint_op(b_kspace)
        r = b - self.forward_backward(x)
        p = r.clone()
        rsold = torch.sum(r * torch.conj(r))
        for i in range(max_iter):
            Ap = self.forward_backward(p)
            alpha = rsold / torch.sum(p * torch.conj(Ap))
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = torch.sum(r * torch.conj(r))
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        if is_complex is False:
            x = torch.stack([x.real, x.imag], dim=1)
        return x, rsold

    def forward(self, input: torch.Tensor, adjoint: bool = False):
        """Run forward or adjoint SENSE operation on the input.

        Depending on if ``adjoint=True``, the input should either be the
        k-space or the complex image. The shapes for these are as follows:
            - kspace: ``(B, H, W, #coils, [2])
            - image: ``(B, H, W, #maps, [2])``

        Args:
            input (torch.Tensor): If ``adjoint=True``, this is the multi-coil k-space,
                else it is the image.
            adjoint (bool, optional): If ``True``, use adjoint operation.

        Returns:
            torch.Tensor: If ``adjoint=True``, the image, else multi-coil k-space.
        """
        if adjoint:
            output = self._adjoint_op(input)
        else:
            output = self._forward_op(input)
        return output


def to_complex(x):
    if not torch.is_complex(x):
        x = torch.complex(x[:, 0], x[:, 1]).unsqueeze(1)
    return x


def to_real(x):
    if torch.is_complex(x):
        x = torch.cat([x.real, x.imag], dim=1)
    return x


class LogLossPlus2(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, clip_tp: float = None, seg_weight=5.0):
        super(LogLossPlus2, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.clip_tp = clip_tp
        self.ddp = ddp
        self.seg_weight = seg_weight

    def forward(self, x, kind, specific_class):
        """
        We want to maximize the loss

        x: the output of the segmentation network
        """
        assert kind in ['upperbound', 'lowerbound'], 'This loss is not designed for standard dice loss'
        x_prob = torch.softmax(x, dim=1)

        assert specific_class is not None
        other_probs = torch.cat([x_prob[:, :specific_class], x_prob[:, specific_class + 1:]], dim=1)
        max_other_probs = other_probs.max(dim=1)[0]
        x_prob = x_prob[:, specific_class]

        if kind == 'upperbound':
            opt_x = x_prob[x_prob < max_other_probs]
            # check if opt_x is empty:
            if opt_x.numel() == 0:
                L = torch.tensor(0)
                return -L
            log1minx = torch.log(1 - opt_x)
            L = log1minx.sum()
        elif kind == 'lowerbound':
            opt_x = x_prob[x_prob > max_other_probs]
            if opt_x.numel() == 0:
                L = torch.tensor(0)
                return -L
            logx = torch.log(opt_x)
            L = logx.sum()

        return -L