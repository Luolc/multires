"""
.. Deep Residual Learning for Image Recognition:
    https://arxiv.org/abs/1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out


class LayerAttention(nn.Module):
    def __init__(self, kdim, expansion=1):
        super().__init__()
        self.kdim = kdim

        shortcuts = dict()
        if expansion != 1:
            shortcuts['64'] = nn.Sequential(
                nn.Conv2d(64, 64 * expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(64 * expansion)
            )
        for channels in [64, 128, 256, 512]:
            shortcuts[str(channels * expansion)] = nn.Sequential(
                nn.Conv2d(channels * expansion, channels * expansion * 2, kernel_size=1, stride=2,
                          bias=False),
                nn.BatchNorm2d(channels * expansion * 2)
            )
        self.shortcuts = nn.ModuleDict(shortcuts)

    def forward(self, query, key_value, residual):
        bsz, kdim = query.size()
        assert kdim == self.kdim
        assert list(query.size()) == [bsz, kdim]
        assert all(list(k.size()) == [bsz, kdim] for k, _ in key_value)

        query = query.view(bsz, kdim, 1)
        # the keys of previous layers with shape (bsz, n_layers, kdim)
        key = torch.stack([k for k, _ in key_value], dim=1)
        attn_weights = torch.bmm(key, query).view(bsz, -1)
        n_layers = len(key_value)
        assert list(attn_weights.size()) == [bsz, n_layers]

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
        attn_weights = attn_weights.view(bsz, n_layers, 1)

        bsz_residual, channels, height, width = residual.size()
        assert bsz_residual == bsz
        assert list(residual.size()) == [bsz, channels, height, width]

        values = []
        for _, v in key_value:
            bsz_value, in_channels, _, _ = v.size()
            assert bsz_value == bsz

            while in_channels != channels:
                v = self.shortcuts[str(in_channels)](v)
                in_channels = v.size(1)
            v = v.view(bsz, -1)
            assert list(v.size()) == [bsz, channels * height * width]

            values.append(v)

        # shape (bsz, channels * height * width, n_layers)
        values = torch.stack(values, dim=-1)

        attn = torch.bmm(values, attn_weights).view(bsz, channels, height, width)

        return attn


class MultiResNet(nn.Module):
    def __init__(self, res_block, n_blocks, kdim, n_classes=10):
        super().__init__()
        self.in_channels = 64

        self.expansion = res_block.expansion
        self.n_blocks = n_blocks

        self.entrance = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer_attn = LayerAttention(kdim, self.expansion)

        projs = dict()
        if self.expansion != 1:
            projs['64'] = nn.Sequential(
                nn.Conv2d(64, 64 * self.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(64 * self.expansion)
            )
        for channels in [64, 128, 256, 512]:
            projs[str(channels * self.expansion)] = nn.Sequential(
                nn.Conv2d(channels * self.expansion, channels * self.expansion * 2,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(channels * self.expansion * 2)
            )
        self.projs = nn.ModuleDict(projs)

        self.q_proj = nn.Linear(512 * self.expansion, kdim)
        self.k_proj = nn.Linear(512 * self.expansion, kdim)

        self.residuals = nn.ModuleList([])
        self.residuals.extend(self.create_res_blocks(res_block, 64, n_blocks[0], stride=1))
        self.residuals.extend(self.create_res_blocks(res_block, 128, n_blocks[1], stride=2))
        self.residuals.extend(self.create_res_blocks(res_block, 256, n_blocks[2], stride=2))
        self.residuals.extend(self.create_res_blocks(res_block, 512, n_blocks[3], stride=2))

        self.out_linear = nn.Linear(512 * self.expansion, n_classes)

    def create_res_blocks(self, res_block, channels, n_blocks, stride):
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(res_block(self.in_channels, channels, stride))
            self.in_channels = channels * self.expansion
        return layers

    def forward(self, x):
        x = self.entrance(x)
        key_value = [(self.proj_k(x), x)]

        for res in self.residuals:
            query = self.proj_q(x)
            residual = res(x)
            x = residual + self.layer_attn(query, key_value, residual)
            x = F.relu(x)
            key_value.append((self.proj_k(x), x))

        x = F.avg_pool2d(x, 4).view(x.size(0), -1)
        x = self.out_linear(x)

        return x

    def projs_x(self, feature_map):
        bsz, in_channels, _, _ = feature_map.size()

        while in_channels != 512 * self.expansion:
            feature_map = self.projs[str(in_channels)](feature_map)
            in_channels = feature_map.size(1)

        out = F.avg_pool2d(feature_map, 4)
        out = out.view(bsz, -1)
        assert list(out.size()) == [bsz, 512 * self.expansion]

        return out

    def proj_k(self, feature_map):
        return self.k_proj(self.projs_x(feature_map))

    def proj_q(self, feature_map):
        return self.q_proj(self.projs_x(feature_map))


def multi_resnet34(kdim=32):
    return MultiResNet(BasicResidualBlock, [3, 4, 6, 3], kdim)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * channels, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion * channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, self.expansion * channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * channels, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion * channels)
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
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels * block.expansion
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


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
