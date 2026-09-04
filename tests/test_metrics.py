"""
Değerlendirme Metrikleri (BLEU, Perplexity, Accuracy) Testleri.
"""

import torch
import pytest
from src import (
    compute_bleu,
    corpus_bleu,
    calculate_perplexity,
    exact_match_accuracy,
    token_accuracy,
)


def test_compute_bleu():
    ref = [1, 2, 3, 4, 5, 6]
    cand_perfect = [1, 2, 3, 4, 5, 6]
    cand_mismatch = [9, 9, 9, 9, 9, 9]

    assert compute_bleu(ref, cand_perfect) == 100.0
    assert compute_bleu(ref, cand_mismatch) == 0.0

    # Corpus BLEU
    refs = [[1, 2, 3, 4], [5, 6, 7, 8]]
    cands = [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert corpus_bleu(refs, cands) == 100.0


def test_calculate_perplexity():
    assert calculate_perplexity(0.0) == 1.0
    assert calculate_perplexity(torch.tensor(1.0)) == 2.718


def test_exact_match_and_token_accuracy():
    preds = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]])
    targets = torch.tensor([[1, 2, 3, 0], [4, 5, 9, 0]])

    # 1. örnek tam eşleşiyor, 2. örnek eşleşmiyor -> 50%
    assert exact_match_accuracy(preds, targets, pad_idx=0) == 50.0

    # Token accuracy: toplam 6 non-pad token, 5 tanesi doğru -> 5/6 = 83.33%
    assert token_accuracy(preds, targets, pad_idx=0) == 83.33
