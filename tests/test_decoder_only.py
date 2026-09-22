"""
Decoder-Only Modeli Test Paketi (tests/test_decoder_only.py).
"""

import pytest
import torch
from src.decoder_only import DecoderOnlyTransformer


def test_decoder_only_forward_and_loss():
    vocab_size = 50
    batch_size = 2
    seq_len = 8

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=32,
    )

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits, loss = model(tokens, targets=targets)

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert loss is not None
    assert loss.item() > 0.0
    assert not torch.isnan(loss)


def test_decoder_only_kv_cache_generation():
    vocab_size = 50
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=32,
    )

    prompt = torch.tensor([[1, 5, 10]], dtype=torch.long)

    # Cache ile üretim
    gen_cached = model.generate(
        prompt,
        max_new_tokens=5,
        temperature=0.0,  # deterministik greedy
        use_cache=True,
    )

    # Cache olmadan üretim
    gen_nocache = model.generate(
        prompt,
        max_new_tokens=5,
        temperature=0.0,
        use_cache=False,
    )

    assert gen_cached.shape == (1, 8)
    assert gen_nocache.shape == (1, 8)
    # Deterministik modda cache'li ve cache'siz aynı sonucu vermelidir!
    assert torch.equal(gen_cached, gen_nocache)
