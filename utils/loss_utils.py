#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import lpips
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp, floor

import numpy as np
import cv2

try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2



class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def reg_l1_loss(network_output, gt):
    return torch.abs((network_output - gt) / (network_output.detach() + 1e-3)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()


def ms_ssim(x, y, scales=3, window_size=11):
    ssim_loss = 0
    for i in range(scales):
        ssim_loss += 1 / scales * ssim(x, y, window_size)
        x = F.avg_pool2d(x, 2, stride=2)
        y = F.avg_pool2d(y, 2, stride=2)

    return ssim_loss


def image2canny(image, thres1, thres2, isEdge1=True):
    """ image: (H, W, 3)"""
    canny_mask = torch.from_numpy(
        cv2.Canny((image.detach().cpu().numpy() * 255.).astype(np.uint8), thres1, thres2) / 255.)
    if not isEdge1:
        canny_mask = 1. - canny_mask
    return canny_mask.float()


with torch.no_grad():
    kernelsize = 3
    conv = torch.nn.Conv2d(1, 1, kernel_size=kernelsize, padding=(kernelsize // 2))
    kernel = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]]).reshape(1, 1, kernelsize, kernelsize)
    conv.weight.data = kernel  # torch.ones((1,1,kernelsize,kernelsize))
    conv.bias.data = torch.tensor([0.])
    conv.requires_grad_(False)
    conv = conv.cuda()


def nearMean_map(array, mask, kernelsize=3):
    """ array: (H,W) / mask: (H,W) """
    cnt_map = torch.ones_like(array)

    nearMean_map = conv((array * mask)[None, None])
    cnt_map = conv((cnt_map * mask)[None, None])
    nearMean_map = (nearMean_map / (cnt_map + 1e-8)).squeeze()

    return nearMean_map


def anisotropic_total_variation_loss(img):
    d_w = torch.abs(img[:, :-1] - img[:, 1:])
    d_h = torch.abs(img[:-1, :] - img[1:, :])
    w_variation = torch.mean(d_w)
    h_variance = torch.mean(d_h)
    return h_variance + w_variation



def depth_aware_smooth_loss(invDepth, rgb_image, depth_mask=None, gamma=0.001):
    """
    Smoothness loss considering depth influence (simplified version, processes 3D tensors)
    For D' = 1/(D + 1), we can derive D = (1/D' - 1)

    Args:
        invDepth: Inverse depth map [1, H, W]
        rgb_image: RGB image [3, H, W]
        depth_mask: Depth mask [1, H, W]
        gamma: Parameter to avoid division by zero
    """
    # Recover depth from inverse depth (for weight calculation)
    depth = (1.0 / invDepth.clamp(min=0.001)) - 1.0

    # Small weights for near regions (small depth), large weights for far regions (large depth)
    depth_weights = torch.exp(-depth / depth.mean())

    # Calculate gradients
    invdepth_dx = invDepth[:, :, 1:] - invDepth[:, :, :-1]  # [1, H, W-1]
    invdepth_dy = invDepth[:, 1:, :] - invDepth[:, :-1, :]  # [1, H-1, W]

    # Propagate weights
    weight_dx = (depth_weights[:, :, 1:] + depth_weights[:, :, :-1]) / 2
    weight_dy = (depth_weights[:, 1:, :] + depth_weights[:, :-1, :]) / 2

    # RGB gradients
    rgb_dx = rgb_image[:, :, 1:] - rgb_image[:, :, :-1]  # [3, H, W-1]
    rgb_dy = rgb_image[:, 1:, :] - rgb_image[:, :-1, :]  # [3, H-1, W]

    # RGB gradient norms
    rgb_dx_norm = torch.sqrt(torch.sum(rgb_dx ** 2, dim=0, keepdim=True) + 1e-8)  # [1, H, W-1]
    rgb_dy_norm = torch.sqrt(torch.sum(rgb_dy ** 2, dim=0, keepdim=True) + 1e-8)  # [1, H-1, W]

    rgb_dx_norm = torch.clamp(rgb_dx_norm, min=gamma)
    rgb_dy_norm = torch.clamp(rgb_dy_norm, min=gamma)

    # Weighted smoothness terms
    smooth_x = weight_dx * torch.abs(invdepth_dx) / rgb_dx_norm
    smooth_y = weight_dy * torch.abs(invdepth_dy) / rgb_dy_norm

    if depth_mask is not None:
        # Create gradient mask
        mask_dx = depth_mask[:, :, 1:] * depth_mask[:, :, :-1]  # [1, H, W-1]
        mask_dy = depth_mask[:, 1:, :] * depth_mask[:, :-1, :]  # [1, H-1, W]

        smooth_x = smooth_x * mask_dx
        smooth_y = smooth_y * mask_dy

        loss = (smooth_x.sum() / (mask_dx.sum() + 1e-8) +
                smooth_y.sum() / (mask_dy.sum() + 1e-8))
    else:
        loss = smooth_x.mean() + smooth_y.mean()

    return loss

def rgb_saturation_loss(rgb, saturation_val: float):
    """
    Calculate RGB saturation loss
    :param rgb: RGB image
    :param saturation_val: Saturation value
    :return: RGB saturation loss
    """
    # Calculate saturation
    max_rgb = torch.max(rgb, dim=1)[0]
    min_rgb = torch.min(rgb, dim=1)[0]
    saturation = (max_rgb - min_rgb) / (max_rgb + 1e-6)
    # Calculate loss
    loss = torch.mean(torch.abs(saturation - saturation_val))
    return loss