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
    def __init__(self, kdim, expansion=1, qk_same=True):
        super().__init__()
        self.kdim = kdim
        self.expansion = expansion
        self.qk_same = qk_same

        shortcuts = dict()
        if expansion != 1:
            shortcuts['16'] = nn.Conv2d(16, 16 * expansion, kernel_size=1, stride=1, bias=False)
        for channels in [16, 32]:
            shortcuts[str(channels * expansion)] = nn.Conv2d(channels * expansion,
                                                             channels * expansion * 2,
                                                             kernel_size=1, stride=2,
                                                             bias=False)
        self.shortcuts = nn.ModuleDict(shortcuts)

        self.batch_norms = nn.ModuleDict({
            str(c): nn.BatchNorm2d(c * expansion) for c in [16, 32, 64]
        })

        self.q_proj = nn.Linear(64 * expansion, kdim)
        self.k_proj = nn.Linear(64 * expansion, kdim)

        if qk_same:
            self.k_proj.weight = self.q_proj.weight

        self.reset_parameters()

    def reset_parameters(self):
        if self.qk_same:
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.constant_(self.q_proj.bias, 0.)
        else:
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)

    def forward(self, query, key_value, residual):
        q = self.proj_q(query)
        bsz, kdim = q.size()
        assert kdim == self.kdim
        assert list(q.size()) == [bsz, kdim]
        q = q.view(bsz, kdim, 1)

        n_layers = len(key_value)
        # the keys of previous layers with shape (bsz, n_layers, kdim)
        key = torch.stack([self.proj_k(k) for k in key_value], dim=1)
        assert list(key.size()) == [bsz, n_layers, kdim]

        attn_weights = torch.bmm(key, q).view(bsz, -1)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
        attn_weights = attn_weights.view(bsz, n_layers, 1)

        bsz_residual, channels, height, width = residual.size()
        assert bsz_residual == bsz
        assert list(residual.size()) == [bsz, channels, height, width]

        values = []
        for v in key_value:
            bsz_value, in_channels, _, _ = v.size()
            assert bsz_value == bsz

            need_batch_norm = False
            while in_channels != channels:
                need_batch_norm = True
                v = self.shortcuts[str(in_channels)](v)
                in_channels = v.size(1)

            if need_batch_norm:
                v = self.batch_norms[str(channels)](v)

            v = v.view(bsz, -1)
            assert list(v.size()) == [bsz, channels * height * width]

            values.append(v)

        # shape (bsz, channels * height * width, n_layers)
        values = torch.stack(values, dim=-1)

        attn = torch.bmm(values, attn_weights).view(bsz, channels, height, width)

        return attn

    def prepare_proj_qk(self, feature_map):
        bsz, channels, _, _ = feature_map.size()

        while channels != 64 * self.expansion:
            feature_map = self.shortcuts[str(channels)](feature_map)
            channels = feature_map.size(1)

        out = F.avg_pool2d(feature_map, 8)
        out = out.view(bsz, -1)
        assert list(out.size()) == [bsz, 64 * self.expansion]

        return out

    def proj_q(self, feature_map):
        return self.q_proj(self.prepare_proj_qk(feature_map))

    def proj_k(self, feature_map):
        return self.k_proj(self.prepare_proj_qk(feature_map))


class MultiResNet(nn.Module):
    def __init__(self, res_block, n_blocks, kdim, n_classes=10, mem_strategy='all'):
        super().__init__()
        assert mem_strategy in ['all', 'one', 'two', 'same_dim']

        self.mem_strategy = mem_strategy
        self.expansion = res_block.expansion

        self.in_channels = 16

        self.entrance = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.layer_attn = LayerAttention(kdim, self.expansion, qk_same=True)

        self.residuals = nn.ModuleList([])
        self.residuals.extend(self.create_res_blocks(res_block, 16, n_blocks[0], stride=1))
        self.residuals.extend(self.create_res_blocks(res_block, 32, n_blocks[1], stride=2))
        self.residuals.extend(self.create_res_blocks(res_block, 64, n_blocks[2], stride=2))

        self.out_linear = nn.Linear(64 * self.expansion, n_classes)

    def create_res_blocks(self, res_block, channels, n_blocks, stride):
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(res_block(self.in_channels, channels, stride))
            self.in_channels = channels * self.expansion
        return layers

    def forward(self, x):
        x = self.entrance(x)
        key_value = [x]

        for res in self.residuals:
            residual = res(x)
            x = residual + self.layer_attn(x, key_value, residual)
            x = F.relu(x)

            if self.mem_strategy == 'all':
                key_value.append(x)
            elif self.mem_strategy == 'one':
                key_value = [x]
            elif self.mem_strategy == 'two':
                key_value = [key_value[-1], x]
            elif self.mem_strategy == 'same_dim':
                # todo
                pass

        x = F.avg_pool2d(x, 8).view(x.size(0), -1)
        x = self.out_linear(x)

        return x


def multi_resnet32(kdim, mem_strategy):
    return MultiResNet(BasicResidualBlock, [5, 5, 5], kdim, mem_strategy=mem_strategy)
