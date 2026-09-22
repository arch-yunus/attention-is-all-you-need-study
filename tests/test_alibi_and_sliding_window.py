"""
ALiBi, Sliding Window Attention ve SDPA Test Paketi.
"""

import pytest
import torch
from src.attention_variants import (
    ALiBiAttention,
    get_alibi_slopes,
    SlidingWindowAttention,
    FlashAttentionSDPA,
)


def test_alibi_slopes_and_forward():
    num_heads = 8
    slopes = get_alibi_slopes(num_heads)
    assert len(slopes) == num_heads
    assert slopes[0] > slopes[-1]  # Eğimler azalan sırada geometrik olmalı

    d_model = 64
    alibi = ALiBiAttention(d_model=d_model, num_heads=num_heads)
    x = torch.randn(2, 10, d_model)
    out, weights = alibi(x, x, x)

    assert out.shape == (2, 10, d_model)
    assert weights.shape == (2, num_heads, 10, 10)
    assert not torch.isnan(out).any()


def test_sliding_window_attention():
    d_model = 64
    num_heads = 4
    window_size = 4
    swa = SlidingWindowAttention(d_model=d_model, num_heads=num_heads, window_size=window_size)

    x = torch.randn(2, 12, d_model)
    out, weights = swa(x)

    assert out.shape == (2, 12, d_model)
    assert weights.shape == (2, num_heads, 12, 12)
    # Pencerenin dışındaki elemanların dikkat ağırlığı 0 olmalıdır (softmax sonrası)
    # i = 10 için j = 0 pencere dışıdır (10 - 0 = 10 >= 4)
    assert torch.all(weights[:, :, 10, 0] < 1e-4)


def test_flash_attention_sdpa():
    d_model = 64
    num_heads = 4
    sdpa = FlashAttentionSDPA(d_model=d_model, num_heads=num_heads)

    x = torch.randn(2, 8, d_model)
    out = sdpa(x, x, x, is_causal=True)

    assert out.shape == (2, 8, d_model)
    assert not torch.isnan(out).any()
