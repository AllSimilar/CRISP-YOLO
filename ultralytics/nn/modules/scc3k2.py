import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv
from .block import C2f, C3, Bottleneck, PSABlock, C3k


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

# ---------- SCConv 三件套（保持原样） ----------
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num, group_num=16, eps=1e-10):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num, f"c_num={c_num} < group_num={group_num}"
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.gamma + self.beta


class SRU(nn.Module):
    def __init__(self, oup_channels, group_num=16, gate_treshold=0.5):
        super().__init__()
        self.gn = GroupBatchnorm2d(oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.gamma / sum(self.gn.gamma)
        reweigts = self.sigomid(gn_x * w_gamma)
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * x
        x_2 = noninfo_mask * x
        return self.reconstruct(x_1, x_2)

    @staticmethod
    def reconstruct(x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    def __init__(self, op_channel, alpha=1/2, squeeze_radio=2,
                 group_size=2, group_kernel_size=3):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, 1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, 1, bias=False)
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel,
                             kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio,
                              op_channel - low_channel // squeeze_radio, kernel_size=1, bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(self, op_channel, group_num=16, gate_treshold=0.5,
                 alpha=1/2, squeeze_radio=2, group_size=2, group_kernel_size=3):
        super().__init__()
        # 防御：group_num 不能超过通道数
        group_num = min(group_num, op_channel)
        self.SRU = SRU(op_channel, group_num=group_num, gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel, alpha=alpha, squeeze_radio=squeeze_radio,
                       group_size=group_size, group_kernel_size=group_kernel_size)

    def forward(self, x):
        return self.CRU(self.SRU(x))

class ScBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=1.0):
        super().__init__()
        c_ = int(c2 * e)
        assert c1 == c_ == c2, (
            f"ScBottleneck 要求 c1==c_==c2（因为 ScConv 通道不变），"
            f"got c1={c1}, c_={c_}, c2={c2}, e={e}"
        )
        self.cv1 = ScConv(c_) # 以前的版本
        # self.cv1 = Conv(c1, c_, k[0], 1) # 现在的版本
        self.cv2 = ScConv(c2)
        self.add = shortcut and (c1 == c2)

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class SCC3k(C3):
    """C3k 变体: 内部用 ScBottleneck 替换标准 Bottleneck"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(ScBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


class SCC3k2(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5,
                 attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                ScBottleneck(self.c, self.c, shortcut, g, e=1.0), 
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else SCC3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else ScBottleneck(self.c, self.c, shortcut, g, e=1.0) 
            for _ in range(n)
        )
