import torch
from torch import nn
from torch.nn import functional as F

from src.model import resnet

def _resnet_bblock_enc_layer_old(inplanes, planes, stride=2, num_blocks=1):
    """Create a ResNet layer for the topdown network specifically"""
    expansion = 1

    downsample = nn.Sequential(
        resnet.conv1x1(inplanes, planes * expansion, stride),
        nn.BatchNorm2d(planes * expansion),
    )
    return nn.Sequential(
        resnet.BasicBlock(inplanes, inplanes),
        resnet.BasicBlock(inplanes, planes, stride, downsample),
    )
class Down_old(nn.Module):
    """Downscaling using residual BasicBlocks with stride of 2"""

    def __init__(self, in_planes, out_planes, num_blocks, stride=2):
        super().__init__()

        # Condense identity channels
        condense = nn.Sequential(
            resnet.conv1x1(in_planes, out_planes, stride=1),
            nn.BatchNorm2d(out_planes),
        )

        # Process and condense channels
        self.conv = nn.Sequential(
            *[
                resnet.BasicBlock(in_planes, out_planes, stride=1, downsample=condense)
                for _ in range(num_blocks - 1)
            ]
        )

        # Process and downsample using stride
        downsample = nn.Sequential(
            resnet.conv1x1(out_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
        )
        self.down = nn.Sequential(
            *[
                resnet.BasicBlock(
                    out_planes, out_planes, stride=stride, downsample=downsample
                )
                for _ in range(1)
            ]
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.down(out)
        return out

class DLA_Node_old(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size=3, padding=1, norm="BatchNorm"
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, padding=padding
        )

        if norm == "GroupNorm":
            self.bn = nn.GroupNorm(out_planes // 8, out_planes)
        elif norm == "BatchNorm":
            self.bn = nn.BatchNorm2d(out_planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class IDA_up(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, padding=1):
        super().__init__()

        self.project = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.ConvTranspose2d(
            out_planes,
            out_planes,
            kernel_size=kernel_size,
            padding=padding,
            stride=2,
        )

    def forward(self, x):
        x = self.project(x)
        x = self.upsample(x)
        return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x