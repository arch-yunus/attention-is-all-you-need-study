"""
Modern Katmanlar ve Aktivasyonlar Test Paketi (tests/test_modern_layers.py).
"""

import pytest
import torch
from src.modern_layers import RMSNorm, SwiGLU, GeGLU, DeepNormScale
from src.residual_norm import LayerNorm


def test_rmsnorm_shape_and_invariance():
    batch_size, seq_len, dim = 2, 8, 64
    x = torch.randn(batch_size, seq_len, dim)
    norm = RMSNorm(dim=dim)

    out = norm(x)
    assert out.shape == (batch_size, seq_len, dim)
    assert not torch.isnan(out).any()

    # Ölçekleme değişmezliği testi: RMSNorm(2 * x) ölçek katsayısını korumalı
    out_scaled = norm(2.0 * x)
    assert torch.allclose(out, out_scaled, atol=1e-5)


def test_swiglu_and_geglu():
    batch_size, seq_len, d_model = 2, 8, 64
    x = torch.randn(batch_size, seq_len, d_model)

    swiglu = SwiGLU(d_model=d_model, d_ff=128)
    geglu = GeGLU(d_model=d_model, d_ff=128)

    out_swiglu = swiglu(x)
    out_geglu = geglu(x)

    assert out_swiglu.shape == (batch_size, seq_len, d_model)
    assert out_geglu.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(out_swiglu).any()
    assert not torch.isnan(out_geglu).any()


def test_deepnorm_scaling():
    scale = DeepNormScale(num_encoder_layers=6, num_decoder_layers=6)
    x = torch.ones(2, 4, 32)
    sublayer_out = torch.ones(2, 4, 32)

    scaled_x = scale(x, sublayer_out)
    assert scaled_x.shape == (2, 4, 32)
    assert torch.all(scaled_x > 1.0)
