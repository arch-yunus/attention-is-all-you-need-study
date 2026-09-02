"""
Katmanlar Arası Tensör Boyut Bütünlüğü Testleri.
"""

import pytest
import torch
from src import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    LayerNorm,
    SublayerConnection,
    EncoderLayer,
    Encoder,
    DecoderLayer,
    Decoder,
    Transformer,
)


def test_scaled_dot_product_attention_shapes():
    batch, heads, seq_len_q, seq_len_k, d_k, d_v = 2, 4, 8, 10, 32, 32
    attn = ScaledDotProductAttention(dropout=0.0)

    q = torch.randn(batch, heads, seq_len_q, d_k)
    k = torch.randn(batch, heads, seq_len_k, d_k)
    v = torch.randn(batch, heads, seq_len_k, d_v)

    out, weights = attn(q, k, v)
    assert out.shape == (batch, heads, seq_len_q, d_v)
    assert weights.shape == (batch, heads, seq_len_q, seq_len_k)
    # Satır toplamı 1 olmalı
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(batch, heads, seq_len_q))


def test_multi_head_attention_shapes():
    batch, seq_len, d_model, num_heads = 3, 12, 64, 4
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)

    x = torch.randn(batch, seq_len, d_model)
    out, weights = mha(x, x, x)

    assert out.shape == (batch, seq_len, d_model)
    assert weights.shape == (batch, num_heads, seq_len, seq_len)


def test_positional_encoding_shapes():
    batch, seq_len, d_model = 2, 25, 128
    pe = PositionalEncoding(d_model=d_model, dropout=0.0, max_len=100)

    x = torch.zeros(batch, seq_len, d_model)
    out = pe(x)

    assert out.shape == (batch, seq_len, d_model)
    # Boş x girdisi pe tensörünün kendisine eşit olmalı
    torch.testing.assert_close(out, pe.pe[:, :seq_len].expand_as(out))


def test_feed_forward_shapes():
    batch, seq_len, d_model, d_ff = 4, 15, 64, 256
    ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.0)

    x = torch.randn(batch, seq_len, d_model)
    out = ffn(x)

    assert out.shape == (batch, seq_len, d_model)


def test_encoder_and_decoder_shapes():
    batch = 2
    src_len = 7
    tgt_len = 5
    d_model = 64
    num_heads = 4
    d_ff = 128

    encoder = Encoder(num_layers=2, d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    decoder = Decoder(num_layers=2, d_model=d_model, num_heads=num_heads, d_ff=d_ff)

    src = torch.randn(batch, src_len, d_model)
    tgt = torch.randn(batch, tgt_len, d_model)

    memory = encoder(src)
    assert memory.shape == (batch, src_len, d_model)

    out = decoder(tgt, memory)
    assert out.shape == (batch, tgt_len, d_model)


def test_transformer_full_model_shapes():
    batch = 2
    src_len = 9
    tgt_len = 8
    src_vocab = 50
    tgt_vocab = 60
    d_model = 64

    model = Transformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=d_model,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
    )

    src = torch.randint(1, src_vocab, (batch, src_len))
    tgt = torch.randint(1, tgt_vocab, (batch, tgt_len))

    logits = model(src, tgt, pad_idx=0)
    assert logits.shape == (batch, tgt_len, tgt_vocab)
