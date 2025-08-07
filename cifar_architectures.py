"""
Collection of CIFAR-specific architectures
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet10(**kwargs):
    return ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


# Helper function for all DP models
def _get_optimal_groups(channels, base_groups=8, strategy="small_groups"):
    """Get optimal number of groups for GroupNorm based on channel count."""
    
    if strategy == "individual":
        # Individual channel normalization (closest to BatchNorm)
        # Each channel gets its own normalization
        return channels
    
    elif strategy == "small_groups":
        # Target 2-4 channels per group for good normalization with some cross-channel interaction
        if channels <= 4:
            return channels
        elif channels <= 8:
            return channels // 2
        else:
            # Target ~4 channels per group
            target_groups = channels // 4
            for groups in range(target_groups, 0, -1):
                if channels % groups == 0:
                    return groups
            return 1
    
    elif strategy == "fixed_8":
        # Fixed 8 groups (original approach)
        for groups in range(min(8, channels), 0, -1):
            if channels % groups == 0:
                return groups
        return 1
    
    else:
        # Default: individual channel normalization
        return channels


# DP-compatible versions using GroupNorm instead of BatchNorm
class BasicBlockDP(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, num_groups=8):
        super(BasicBlockDP, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(_get_optimal_groups(planes, num_groups), planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.gn2 = nn.GroupNorm(_get_optimal_groups(planes, num_groups), planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            expansion_channels = self.expansion * planes
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    expansion_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(_get_optimal_groups(expansion_channels, num_groups), expansion_channels),
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + self.shortcut(x)  # Changed from += to + to avoid in-place modification
        out = F.relu(out)
        return out


class BottleneckDP(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, num_groups=8):
        super(BottleneckDP, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(_get_optimal_groups(planes, num_groups), planes)
        
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.gn2 = nn.GroupNorm(_get_optimal_groups(planes, num_groups), planes)
        
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        expansion_channels = self.expansion * planes
        self.gn3 = nn.GroupNorm(_get_optimal_groups(expansion_channels, num_groups), expansion_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(_get_optimal_groups(expansion_channels, num_groups), expansion_channels),
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = F.relu(self.gn2(self.conv2(out)))
        out = self.gn3(self.conv3(out))
        out = out + self.shortcut(x)  # Changed from += to + to avoid in-place modification
        out = F.relu(out)
        return out


class ResNetDP(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_groups=8):
        super(ResNetDP, self).__init__()
        self.in_planes = 64
        self.num_groups = num_groups

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(_get_optimal_groups(64, num_groups), 64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.num_groups))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet10DP(**kwargs):
    return ResNetDP(BasicBlockDP, [1, 1, 1, 1], **kwargs)


def ResNet18DP(**kwargs):
    return ResNetDP(BasicBlockDP, [2, 2, 2, 2], **kwargs)


def ResNet34DP(**kwargs):
    return ResNetDP(BasicBlockDP, [3, 4, 6, 3], **kwargs)


def ResNet50DP(**kwargs):
    return ResNetDP(BottleneckDP, [3, 4, 6, 3], **kwargs)


def ResNet101DP(**kwargs):
    return ResNetDP(BottleneckDP, [3, 4, 23, 3], **kwargs)


def ResNet152DP(**kwargs):
    return ResNetDP(BottleneckDP, [3, 8, 36, 3], **kwargs)


class ResNetExtraInputs(nn.Module):
    def __init__(
        self, block, num_blocks, num_classes=10, hidden_dims=[], extra_inputs=None
    ):
        super(ResNetExtraInputs, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        prev_size = 512 * block.expansion
        if extra_inputs is not None:
            prev_size += extra_inputs
        mlp_list = []
        for hd in hidden_dims:
            mlp_list.append(torch.nn.Linear(prev_size, hd))
            mlp_list.append(torch.nn.LeakyReLU())
            prev_size = hd
        mlp_list.append(torch.nn.Linear(prev_size, num_classes))
        self.linear = torch.nn.Sequential(*mlp_list)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, extra_inputs=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if extra_inputs is not None:
            assert (
                extra_inputs.shape[0] == out.shape[0] and extra_inputs.ndim == out.ndim
            )
            out = torch.concatenate([out, extra_inputs], dim=1)
        out = self.linear(out)
        return out


def ResNet10ExtraInputs(**kwargs):
    return ResNetExtraInputs(BasicBlock, [1, 1, 1, 1], **kwargs)


def ResNet18ExtraInputs(**kwargs):
    return ResNetExtraInputs(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet34ExtraInputs(**kwargs):
    return ResNetExtraInputs(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet50ExtraInputs(**kwargs):
    return ResNetExtraInputs(Bottleneck, [3, 4, 6, 3], **kwargs)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


# Wide ResNets in the style of https://objax.readthedocs.io/en/latest/_modules/objax/zoo/wide_resnet.html#WRNBlock

BN_MOM = 0.9
BN_EPS = 1e-5


class WRNBlock(nn.Module):
    """WideResNet block."""

    def __init__(
        self, nin: int, nout: int, stride: int = 1, bn: Callable = nn.BatchNorm2d
    ):
        """Creates WRNBlock instance.

        Args:
            nin: number of input filters.
            nout: number of output filters.
            stride: stride for convolution and projection convolution in this block.
            bn: module which used as batch norm function.
        """
        super().__init__()
        if nin != nout or stride > 1:
            # self.proj_conv = objax.nn.Conv2D(nin, nout, 1, strides=stride, **conv_args(1, nout))
            self.proj_conv = nn.Conv2d(
                nin, nout, kernel_size=1, stride=stride, bias=False
            )
        else:
            self.proj_conv = None

        # Handle different normalization layer parameters
        # Check if it's BatchNorm (which takes eps and momentum) or other norm layers
        if 'BatchNorm' in str(bn):
            # BatchNorm or other normalization that accepts eps and momentum
            self.norm_1 = bn(nin, eps=BN_EPS, momentum=BN_MOM)
            self.norm_2 = bn(nout, eps=BN_EPS, momentum=BN_MOM)
        else:
            # GroupNorm or other normalization that doesn't take eps/momentum
            self.norm_1 = bn(nin)
            self.norm_2 = bn(nout)
            
        # self.conv_1 = objax.nn.Conv2D(nin, nout, 3, strides=stride, **conv_args(3, nout))
        self.conv_1 = nn.Conv2d(
            nin, nout, kernel_size=3, stride=stride, bias=False, padding=1
        )
        # self.conv_2 = objax.nn.Conv2D(nout, nout, 3, strides=1, **conv_args(3, nout))
        self.conv_2 = nn.Conv2d(
            nout, nout, kernel_size=3, stride=1, bias=False, padding=1
        )

    def forward(self, x):
        o1 = F.relu(self.norm_1(x))
        y = self.conv_1(o1)
        o2 = F.relu(self.norm_2(y))
        z = self.conv_2(o2)
        return z + self.proj_conv(o1) if self.proj_conv else z + x


class WideResNetGeneral(nn.Module):
    """Base WideResNet implementation."""

    def __init__(
        self,
        nin: int,
        nclass: int,
        blocks_per_group: List[int],
        width: int,
        bn: Callable = nn.BatchNorm2d,
    ):
        """Creates WideResNetGeneral instance.

        Args:
            nin: number of channels in the input image.
            nclass: number of output classes.
            blocks_per_group: number of blocks in each block group.
            width: multiplier to the number of convolution filters.
            bn: module which used as batch norm function.
        """
        super().__init__()
        widths = [
            int(v * width)
            for v in [16 * (2**i) for i in range(len(blocks_per_group))]
        ]

        n = 16
        # ops = [objax.nn.Conv2D(nin, n, 3, **conv_args(3, n))]
        ops = [nn.Conv2d(nin, n, kernel_size=3, bias=False, padding=1)]
        for i, (block, width) in enumerate(zip(blocks_per_group, widths)):
            stride = 2 if i > 0 else 1
            ops.append(WRNBlock(n, width, stride, bn))
            for b in range(1, block):
                ops.append(WRNBlock(width, width, 1, bn))
            n = width
        ops += [
            # Handle the final normalization layer - GroupNorm doesn't take eps/momentum
            bn(n, eps=BN_EPS, momentum=BN_MOM) if 'BatchNorm' in str(bn) else bn(n),
            # objax.functional.relu,
            nn.ReLU(),
            # self.mean_reduce,
            torch.nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # objax.nn.Linear(n, nclass, w_init=objax.nn.init.xavier_truncated_normal)
            nn.Linear(n, nclass),
        ]
        self.model = nn.Sequential(*ops)

    def forward(self, x):
        return self.model(x)


class WideResNet(WideResNetGeneral):
    """WideResNet implementation with 3 groups.

    Reference:
        http://arxiv.org/abs/1605.07146
        https://github.com/szagoruyko/wide-residual-networks
    """

    def __init__(
        self,
        num_classes: int = 10,
        nin: int = 3,
        depth: int = 28,
        width: int = 2,
        # bn: Callable = functools.partial(objax.nn.BatchNorm2D, momentum=BN_MOM, eps=BN_EPS)):
        bn: Callable = nn.BatchNorm2d,
    ):
        """Creates WideResNet instance.

        Args:
            nin: number of channels in the input image.
            nclass: number of output classes.
            depth: number of convolution layers. (depth-4) should be divisible by 6
            width: multiplier to the number of convolution filters.
            bn: module which used as batch norm function.
        """
        assert (depth - 4) % 6 == 0, "depth should be 6n+4"
        n = (depth - 4) // 6
        blocks_per_group = [n] * 3
        super().__init__(nin, num_classes, blocks_per_group, width, bn)


class BottleneckNoNorm(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleneckNoNorm, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=True)  # Enable bias since no norm
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
        )
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=True
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=True,
                ),
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetNoNorm(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetNoNorm, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        
        # Apply careful initialization for networks without normalization
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use He initialization with fan_out mode for ReLU networks
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet50NoNorm(**kwargs):
    return ResNetNoNorm(BottleneckNoNorm, [3, 4, 6, 3], **kwargs)
