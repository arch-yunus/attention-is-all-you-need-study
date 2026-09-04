"""
Modern Dikkat Varyantları (MQA, GQA, RoPE) Testleri.
"""

import torch
import pytest
from src import (
    MultiQueryAttention,
    GroupedQueryAttention,
    RotaryPositionalEmbedding,
    compare_attention_memory,
)


def test_multi_query_attention():
    batch, seq_q, seq_k, d_model, num_heads = 2, 8, 8, 64, 4
    mqa = MultiQueryAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)

    q = torch.randn(batch, seq_q, d_model)
    k = torch.randn(batch, seq_k, d_model)
    v = torch.randn(batch, seq_k, d_model)

    out, attn = mqa(q, k, v)
    assert out.shape == (batch, seq_q, d_model)
    assert attn.shape == (batch, num_heads, seq_q, seq_k)


def test_grouped_query_attention():
    batch, seq_q, seq_k, d_model, num_heads, num_kv = 2, 6, 6, 64, 4, 2
    gqa = GroupedQueryAttention(
        d_model=d_model, num_heads=num_heads, num_kv_heads=num_kv, dropout=0.0
    )

    q = torch.randn(batch, seq_q, d_model)
    k = torch.randn(batch, seq_k, d_model)
    v = torch.randn(batch, seq_k, d_model)

    out, attn = gqa(q, k, v)
    assert out.shape == (batch, seq_q, d_model)
    assert attn.shape == (batch, num_heads, seq_q, seq_k)


def test_rotary_positional_embedding():
    batch, num_heads, seq_len, d_k = 2, 4, 10, 16
    rope = RotaryPositionalEmbedding(dim=d_k, max_seq_len=64)

    q = torch.randn(batch, num_heads, seq_len, d_k)
    k = torch.randn(batch, num_heads, seq_len, d_k)

    q_rot, k_rot = rope(q, k)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    # RoPE normu korur (uniter rotasyon matrisi)
    assert torch.allclose(torch.norm(q, dim=-1), torch.norm(q_rot, dim=-1), atol=1e-4)


def test_compare_attention_memory():
    savings = compare_attention_memory(
        batch_size=1,
        seq_len=2048,
        num_heads=32,
        d_model=4096,
        num_layers=32,
    )
    assert savings["MHA_MB"] > savings["GQA_MB"] > savings["MQA_MB"]
    assert savings["GQA_Savings_Pct"] == 75.0
    assert savings["MQA_Savings_Pct"] == 96.9
