"""
Üretim ve Kod Çözme Algoritmaları Testleri.
"""

import torch
import pytest
from src import (
    Transformer,
    greedy_decode,
    beam_search_decode,
    sample_decode,
    apply_sampling_filters,
)


@pytest.fixture
def dummy_model():
    torch.manual_seed(42)
    model = Transformer(
        src_vocab_size=30,
        tgt_vocab_size=30,
        d_model=32,
        num_heads=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=64,
        dropout=0.0,
    )
    model.eval()
    return model


def test_greedy_decode_with_and_without_cache(dummy_model):
    torch.manual_seed(42)
    src = torch.randint(3, 30, (2, 6))

    # KV-Cache kullanarak ve kullanmayarak üret
    out_cached = greedy_decode(dummy_model, src, max_len=8, use_cache=True)
    out_uncached = greedy_decode(dummy_model, src, max_len=8, use_cache=False)

    assert out_cached.shape == out_uncached.shape
    assert torch.equal(out_cached, out_uncached), "KV-Cache ile önbelleksiz çıkarım aynı sonucu üretmelidir!"


def test_beam_search_decode(dummy_model):
    torch.manual_seed(42)
    src = torch.randint(3, 30, (2, 5))

    out_beam = beam_search_decode(dummy_model, src, beam_size=3, max_len=7)
    assert out_beam.size(0) == 2
    assert out_beam.size(1) <= 7
    assert out_beam[0, 0].item() == 1  # BOS token


def test_sample_decode(dummy_model):
    torch.manual_seed(42)
    src = torch.randint(3, 30, (2, 5))

    out_sample = sample_decode(
        dummy_model,
        src,
        max_len=8,
        temperature=0.7,
        top_k=5,
        top_p=0.9,
    )
    assert out_sample.size(0) == 2
    assert out_sample.size(1) <= 8


def test_apply_sampling_filters():
    torch.manual_seed(42)
    logits = torch.tensor([[1.0, 5.0, 2.0, 4.0, 3.0]])

    # Top-K = 2
    filtered_k = apply_sampling_filters(logits.clone(), top_k=2)
    # En büyük iki eleman: indeks 1 (5.0) ve indeks 3 (4.0). Diğerleri -inf olmalı.
    assert filtered_k[0, 0] == float("-inf")
    assert filtered_k[0, 2] == float("-inf")
    assert filtered_k[0, 4] == float("-inf")
    assert filtered_k[0, 1] == 5.0
    assert filtered_k[0, 3] == 4.0

    # Repetition penalty
    prev_tokens = torch.tensor([[1]])
    filtered_rep = apply_sampling_filters(logits.clone(), repetition_penalty=2.0, prev_tokens=prev_tokens)
    assert filtered_rep[0, 1] == 2.5
