import torch

def simple_color_balance(image, scale=0.01, alpha=0.1):
    """
    White balance processing for RGB image tensor in [3, H, W] format, maintaining float precision.

    Parameters:
    image: torch.Tensor, shape=[3, H, W], values in [0, 1]
    scale: saturation clipping ratio (default 0.01)
    alpha: contrast expansion factor (default 0)

    Returns:
    processed image, shape=[3, H, W], value range [0, 1]
    """
    assert image.ndim == 3 and image.shape[0] == 3, "Input must be in [3, H, W] format"

    r, g, b = image[0], image[1], image[2]
    avg_rgb = torch.tensor([r.mean(), g.mean(), b.mean()], device=image.device)
    gray_value = avg_rgb.mean()
    scale_value = gray_value / (avg_rgb + 1e-6)
    sat_level = torch.clamp(scale * scale_value, 0.0, 1.0)

    out = torch.empty_like(image)

    for ch in range(3):
        channel = image[ch].flatten()
        q_low = torch.quantile(channel, sat_level[ch].item())
        q_high = torch.quantile(channel, 1 - sat_level[ch].item())
        clipped = torch.clamp(channel, q_low, q_high)

        var = clipped.std()
        pmin = clipped.min() - alpha * var
        pmax = clipped.max() + alpha * var

        stretched = (clipped - pmin) / (pmax - pmin + 1e-6)
        out[ch] = stretched.reshape(image.shape[1], image.shape[2])

    return out 