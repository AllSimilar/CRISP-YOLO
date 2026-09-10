from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Converse2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        scale: int = 1,
        padding: int = 4,
        padding_mode: str = "circular",
        eps: float = 1e-5,
    ):
        super().__init__()
        assert out_channels == in_channels, (
            f"Converse2D is depthwise: in_channels ({in_channels}) must == out_channels ({out_channels})"
        )
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.scale = scale
        self.padding = padding
        self.padding_mode = padding_mode
        self.eps = eps

        self.weight = nn.Parameter(torch.randn(1, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        with torch.no_grad():
            self.weight.data = F.softmax(self.weight.data.view(1, in_channels, -1), dim=-1).view(
                1, in_channels, kernel_size, kernel_size
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        s = self.scale

        if self.padding > 0:
            x = F.pad(x, [self.padding] * 4, mode=self.padding_mode)

        lam = torch.sigmoid(self.bias - 9.0) + self.eps

        _, _, h, w = x.shape

        x0 = x if s == 1 else F.interpolate(x, scale_factor=float(s), mode="nearest")
        STy = self._sfold_upsample(x, s)

        FK = self._psf_to_otf(self.weight, (h * s, w * s))
        FK_conj = torch.conj(FK)  # F̄_K
        FK_sq = torch.abs(FK).pow(2)  # |F_K|²

        FBFy = FK_conj * torch.fft.fftn(STy, dim=(-2, -1))
        L = FBFy + torch.fft.fftn(lam * x0, dim=(-2, -1))  # L = F̄_K·F_{Y↑s} + λ·F_{X₀}

        FKL = FK * L  # F_K · L
        FKL_s = torch.mean(self._distinct_blocks(FKL, s), dim=-1)  # 分块平均 ⇓_s
        FK2_s = torch.mean(self._distinct_blocks(FK_sq, s), dim=-1)

        Fdiv = FKL_s / (FK2_s + lam)  # (F_K·L)⇓_s / (|F_K|²⇓_s + λ)
        Fmul = FK_conj * Fdiv.repeat(1, 1, s, s)  # F̄_K ⊙_s (...)
        FX = (L - Fmul) / lam

        out = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

        if self.padding > 0:
            p = self.padding * s
            out = out[..., p:-p, p:-p]

        return out

    @staticmethod
    def _sfold_upsample(x: torch.Tensor, scale: int) -> torch.Tensor:
        if scale == 1:
            return x
        B, C, H, W = x.shape
        z = x.new_zeros(B, C, H * scale, W * scale)
        z[..., ::scale, ::scale] = x
        return z

    @staticmethod
    def _distinct_blocks(a: torch.Tensor, scale: int) -> torch.Tensor:
        if scale == 1:
            return a.unsqueeze(-1)
        *leading, W, H = a.size()
        W_s, H_s = W // scale, H // scale
        b = a.view(*leading, scale, W_s, scale, H_s)
        n = len(leading)
        b = b.permute(*range(n), n + 1, n + 3, n, n + 2).contiguous()
        return b.view(*leading, W_s, H_s, scale * scale)

    @staticmethod
    def _psf_to_otf(psf: torch.Tensor, shape: tuple) -> torch.Tensor:
        otf = psf.new_zeros(psf.shape[:-2] + shape)
        otf[..., : psf.shape[-2], : psf.shape[-1]] = psf
        otf = torch.roll(otf, (-psf.shape[-2] // 2, -psf.shape[-1] // 2), dims=(-2, -1))
        return torch.fft.fftn(otf, dim=(-2, -1))


class Decoder(nn.Module):
    def __init__(
        self, ch: tuple = (256, 512, 1024), c2: int = 3, up_to_input: bool = True, converse_final: bool = False
    ):
        super().__init__()
        c_p3, c_p4, c_p5 = ch
        self.up_to_input = up_to_input

        self.reduce_1 = nn.Sequential(
            nn.Conv2d(c_p5, c_p4, 1, bias=False),
            nn.BatchNorm2d(c_p4),
            nn.SiLU(inplace=True),
        )
        self.up_1 = Converse2D(
            c_p4,
            c_p4,
            kernel_size=5,
            scale=2,
            padding=4,
            padding_mode="circular",
            eps=1e-3,
        )

        self.reduce_2 = nn.Sequential(
            nn.Conv2d(2 * c_p4, c_p3, 1, bias=False),
            nn.BatchNorm2d(c_p3),
            nn.SiLU(inplace=True),
        )
        self.up_2 = Converse2D(
            c_p3,
            c_p3,
            kernel_size=5,
            scale=2,
            padding=4,
            padding_mode="circular",
            eps=1e-3,
        )

        c_mid = max(c_p3 // 4, 16)
        self.reduce_3 = nn.Sequential(
            nn.Conv2d(2 * c_p3, c_mid, 1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True),
        )
        self.up_3 = Converse2D(
            c_mid,
            c_mid,
            kernel_size=5,
            scale=2,
            padding=4,
            padding_mode="circular",
            eps=1e-3,
        )

        if not up_to_input:
            self.upsample = nn.Identity()
        elif converse_final:
            self.upsample = Converse2D(
                c_mid,
                c_mid,
                kernel_size=5,
                scale=4,
                padding=4,
                padding_mode="circular",
                eps=1e-3,
            )
        else:
            self.upsample = nn.Upsample(
                scale_factor=4,
                mode="bilinear",
                align_corners=True,
            )

        self.conv_4 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(c_mid, c2, kernel_size=7),
            nn.Tanh(),
        )

    def forward(self, x):
        p3, p4, p5 = x  # high-res → low-res

        c1 = self.reduce_1(p5)
        c1 = self.up_1(c1)
        c1 = torch.cat((c1, p4), dim=1)

        c2 = self.reduce_2(c1)
        c2 = self.up_2(c2)
        c2 = torch.cat((c2, p3), dim=1)

        c3 = self.reduce_3(c2)
        c3 = self.up_3(c3)

        out = self.upsample(c3)

        return self.conv_4(out)


class TransposeDecoder(nn.Module):
    def __init__(
        self, ch: tuple = (256, 512, 1024), c2: int = 3, up_to_input: bool = True, transpose_final: bool = False
    ):
        super().__init__()
        c_p3, c_p4, c_p5 = ch
        self.up_to_input = up_to_input

        self.reduce_1 = nn.Sequential(
            nn.Conv2d(c_p5, c_p4, 1, bias=False),
            nn.BatchNorm2d(c_p4),
            nn.SiLU(inplace=True),
        )
        self.up_1 = nn.ConvTranspose2d(c_p4, c_p4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.reduce_2 = nn.Sequential(
            nn.Conv2d(2 * c_p4, c_p3, 1, bias=False),
            nn.BatchNorm2d(c_p3),
            nn.SiLU(inplace=True),
        )
        self.up_2 = nn.ConvTranspose2d(c_p3, c_p3, kernel_size=3, stride=2, padding=1, output_padding=1)

        c_mid = max(c_p3 // 4, 16)
        self.reduce_3 = nn.Sequential(
            nn.Conv2d(2 * c_p3, c_mid, 1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True),
        )
        self.up_3 = nn.ConvTranspose2d(c_mid, c_mid, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.upsample = (
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True) if up_to_input else nn.Identity()
        )

        self.conv_4 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(c_mid, c2, kernel_size=7),
            nn.Tanh(),
        )

    def forward(self, x):
        p3, p4, p5 = x

        c1 = self.reduce_1(p5)
        c1 = self.up_1(c1)
        c1 = torch.cat((c1, p4), dim=1)

        c2 = self.reduce_2(c1)
        c2 = self.up_2(c2)
        c2 = torch.cat((c2, p3), dim=1)
        c3 = self.reduce_3(c2)
        c3 = self.up_3(c3)

        out = self.upsample(c3)
        return self.conv_4(out)
