from typing import Tuple

import torch


def encode_text(tokens: list[str], dim: int = 128) -> torch.Tensor:
    vec = torch.zeros(dim)
    for t in tokens:
        h = hash(t) % dim
        vec[h] += 1.0
    return vec


def encode_image(img: torch.Tensor, dim: int = 128) -> torch.Tensor:
    return img.flatten()[:dim].float().mean() * torch.ones(dim)


def encode_audio(wave: torch.Tensor, dim: int = 128) -> torch.Tensor:
    spec = torch.fft.rfft(wave)
    return spec.abs()[:dim].mean() * torch.ones(dim)


def encode_video(frames: torch.Tensor, dim: int = 128) -> torch.Tensor:
    return frames.float().mean() * torch.ones(dim)


def poisson_encode(x: torch.Tensor, rate_hz: float, T: int, dt_ms: float = 1.0) -> torch.Tensor:
    dt = dt_ms / 1000.0
    lam = torch.clamp(x.abs() * rate_hz * dt, 0.0, 1.0)
    return torch.bernoulli(lam.expand(T, *x.shape))

