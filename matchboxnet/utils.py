import torch
import torch.nn as nn

def calc_same_padding(kernel_size, stride=1, dilation=1):
    """Compute symmetric padding for 1D convolutions to approximate 'same' output length."""
    padding = (dilation * (kernel_size - 1) + stride - 1) / 2
    return int(padding + 0.5)



class TCSConv(nn.Module):
    """
    Time-Channel Separable 1D Convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size of the convolution.
        stride (int, optional): Stride of the convolution. Default = 1.
        dilation (int, optional): Dilation of the convolution. Default = 1.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        padding = calc_same_padding(kernel_size, stride, dilation)
        self.depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, groups=in_channels, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        """Forward pass."""
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class SubBlock(nn.Module):
    """
    MatchboxNet sub-block: TCSConv -> BN -> ReLU -> Dropout (+ Residual if last in block)

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size of the convolution.
        dilation (int, optional): Dilation for the convolution. Default = 1.
        dropout (float, optional): Dropout rate. Default = 0.2.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dropout=0.2):
        super().__init__()
        self.tcs_conv = TCSConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, residual=None):
        """Forward pass with optional residual connection."""
        x = self.tcs_conv(x)
        x = self.bnorm(x)
        if residual is not None:
            x = x + residual
        x = self.relu(x)
        return self.dropout(x)

class MainBlock(nn.Module):
    """
    A MatchboxNet residual block consisting of R SubBlocks with optional projection for residuals.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size for the sub-blocks.
        R (int, optional): Number of sub-blocks. Default = 3.
        dilation (int, optional): Dilation for the convolutions. Default = 1.
    """
    def __init__(self, in_channels, out_channels, kernel_size, R=3, dilation=1):
        super().__init__()
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else None
        self.residual_bn = nn.BatchNorm1d(out_channels) if in_channels != out_channels else None

        self.sub_blocks = nn.ModuleList()
        self.sub_blocks.append(SubBlock(in_channels, out_channels, kernel_size, dilation))
        for _ in range(1, R):
            self.sub_blocks.append(SubBlock(out_channels, out_channels, kernel_size, dilation))

    def forward(self, x):
        """Forward pass through R SubBlocks with residual connection."""
        residual = x
        if self.residual_conv:
            residual = self.residual_conv(residual)
        if self.residual_bn:
            residual = self.residual_bn(residual)
        for i, sub_block in enumerate(self.sub_blocks):
            x = sub_block(x, residual) if i == len(self.sub_blocks) - 1 else sub_block(x)
        return x
    
    
class MatchboxNet(nn.Module):
    """
    Implementation of MatchboxNet architecture based on:
    https://arxiv.org/pdf/2004.08531

    Args:
        B (int): Number of main blocks.
        R (int): Number of sub-blocks per main block.
        C (int): Number of channels inside the main blocks.
        kernel_sizes (list of int): Kernel sizes for each block.
        num_classes (int): Number of output classes.
        input_channels (int): Number of input feature channels (e.g., MFCCs).
    """
    def __init__(self, B=3, R=2, C=64, kernel_sizes=None, num_classes=30, input_channels=64):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [11 + 2*i for i in range(B)]
            
        elif len(kernel_sizes) != B :
            raise IndexError(f"Kernel sizes length must be equal to {B}")

        self.prologue = nn.Sequential(
            nn.Conv1d(input_channels, 128, 11, stride=2, padding=calc_same_padding(11, stride=2)),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        self.blocks.append(MainBlock(128, C, kernel_sizes[0], R))
        for i in range(1, B):
            self.blocks.append(MainBlock(C, C, kernel_sizes[i], R))

        self.epi_conv1 = TCSConv(C, 128, 29, dilation=2)
        self.epi_bn1 = nn.BatchNorm1d(128)
        self.epi_relu1 = nn.ReLU()

        self.epi_conv2 = TCSConv(128, 128, 1)
        self.epi_bn2 = nn.BatchNorm1d(128)
        self.epi_relu2 = nn.ReLU()

        self.final_conv = nn.Conv1d(128, num_classes, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """Forward pass for classification."""
        x = self.prologue(x)
        for block in self.blocks:
            x = block(x)
        x = self.epi_conv1(x)
        x = self.epi_bn1(x)
        x = self.epi_relu1(x)

        x = self.epi_conv2(x)
        x = self.epi_bn2(x)
        x = self.epi_relu2(x)

        x = self.final_conv(x)
        x = self.pool(x)
        return x.squeeze(2)
    
def symmetric_pad_or_truncate(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Symmetrically pad or truncate the time dimension of a 2D feature tensor.

    This utility ensures that the temporal length of the input features
    matches the required fixed length by adding zero-padding equally on both sides
    or by centrally cropping the tensor.

    Args:
        x (torch.Tensor of shape (C, T)): Input feature tensor, where C is
            the number of channels and T is the current temporal length.
        target_len (int): Desired temporal length after padding/truncation.

    Returns:
        torch.Tensor of shape (C, target_len): Tensor with adjusted time dimension.
    """
    T = x.shape[-1]
    if T < target_len:
        pad = target_len - T
        left = pad // 2
        right = pad - left
        return torch.nn.functional.pad(x, (left, right))
    if T > target_len:
        start = (T - target_len) // 2
        return x[:, start: start + target_len]
    return x