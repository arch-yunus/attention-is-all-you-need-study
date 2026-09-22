"""
Gelişmiş Örnekleme ve Min-P Filtreleme Test Paketi (tests/test_sampling_advanced.py).
"""

import pytest
import torch
from src.generation import apply_sampling_filters


def test_min_p_filter():
    # Vocab boyutu = 5
    logits = torch.tensor([[10.0, 8.0, 2.0, -5.0, -10.0]])
    # min_p = 0.5 uygulandığında çok düşük logitler -inf olmalı
    filtered = apply_sampling_filters(logits=logits.clone(), min_p=0.5)

    assert not torch.isinf(filtered[0, 0])  # En yüksek korunmalı
    assert torch.isneginf(filtered[0, 3])   # Düşük olan -inf olmalı
    assert torch.isneginf(filtered[0, 4])


def test_repetition_and_frequency_penalty():
    logits = torch.tensor([[5.0, 5.0, 5.0]])
    prev_tokens = torch.tensor([[0, 0, 1]])

    # Token 0 iki kez, Token 1 bir kez geçmiş, Token 2 geçmemiş
    filtered = apply_sampling_filters(
        logits=logits.clone(),
        repetition_penalty=2.0,
        frequency_penalty=1.0,
        prev_tokens=prev_tokens,
    )

    # Cezalandırılmayan Token 2, Token 0'dan büyük olmalı
    assert filtered[0, 2] > filtered[0, 0]
    assert filtered[0, 2] > filtered[0, 1]
