"""
Değerlendirme Metrikleri Modülü (Evaluation Metrics).

Transformer modellerinin başarısını ölçmek için kullanılan standart metrikler:
1. BLEU Skoru (Bilingual Evaluation Understudy - Papineni et al., 2002)
2. Karmaşıklık (Perplexity / PPL)
3. Tam Eşleşme Başarımı (Exact Match Accuracy)
4. Token Doğruluk Oranı (Token-level Accuracy)
"""

import math
from collections import Counter
from typing import List, Sequence, Union
import torch


def get_ngrams(tokens: Sequence[int], n: int) -> Counter:
    """Belirli bir n uzunluğundaki n-gram frekanslarını çıkarır."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(
    reference: List[int],
    candidate: List[int],
    max_order: int = 4,
    smooth: bool = True,
) -> float:
    """
    Tekil bir aday ve referans dizisi için BLEU skorunu hesaplar (0.0 ile 100.0 arasında).

    Args:
        reference (List[int]): Referans hedef token listesi.
        candidate (List[int]): Model tarafından üretilen token listesi.
        max_order (int): Azami n-gram derecesi (Varsayılan: 4).
        smooth (bool): Sıfır eşleşme durumunda Chen & Cherry yumuşatması uygula.

    Returns:
        float: BLEU skoru (0 - 100).
    """
    ref_len = len(reference)
    cand_len = len(candidate)

    if cand_len == 0:
        return 0.0

    # 1. Brevity Penalty (Kısalık Cezası)
    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (float(ref_len) / float(cand_len)))

    # 2. Modifiye Edilmiş Hassasiyetler (Modified n-gram precisions)
    precisions = []
    # 1-gram kontrolü
    cand_1grams = get_ngrams(candidate, 1)
    ref_1grams = get_ngrams(reference, 1)
    clipped_1 = sum(min(count, ref_1grams.get(ngram, 0)) for ngram, count in cand_1grams.items())
    if clipped_1 == 0:
        return 0.0

    precisions.append(float(clipped_1) / float(len(candidate)))

    for n in range(2, max_order + 1):
        cand_ngrams = get_ngrams(candidate, n)
        ref_ngrams = get_ngrams(reference, n)

        total_cand = sum(cand_ngrams.values())
        if total_cand == 0:
            if smooth:
                precisions.append(1.0 / (2.0 ** n))
            else:
                precisions.append(0.0)
            continue

        clipped_matches = sum(min(count, ref_ngrams.get(ngram, 0)) for ngram, count in cand_ngrams.items())

        if clipped_matches == 0:
            if smooth:
                precisions.append(1.0 / (2.0 * total_cand))
            else:
                precisions.append(0.0)
        else:
            precisions.append(float(clipped_matches) / float(total_cand))

    if min(precisions) <= 0.0:
        return 0.0

    # Geometrik Ortalama
    log_sum = sum(math.log(p) for p in precisions) / float(max_order)
    bleu = bp * math.exp(log_sum) * 100.0
    return round(bleu, 2)


def corpus_bleu(
    references: List[List[int]],
    candidates: List[List[int]],
    max_order: int = 4,
    smooth: bool = True,
) -> float:
    """Tüm korpus üzerindeki ortalama BLEU skorunu hesaplar."""
    if not references or not candidates or len(references) != len(candidates):
        return 0.0
    scores = [
        compute_bleu(ref, cand, max_order=max_order, smooth=smooth)
        for ref, cand in zip(references, candidates)
    ]
    return round(sum(scores) / len(scores), 2)


def calculate_perplexity(loss: Union[float, torch.Tensor]) -> float:
    """
    Çapraz Entropi (Cross-Entropy) kaybından Perplexity (PPL) değerini hesaplar.
    PPL = exp(Cross-Entropy Loss)
    """
    if isinstance(loss, torch.Tensor):
        loss_val = loss.item()
    else:
        loss_val = float(loss)

    try:
        ppl = math.exp(loss_val)
    except OverflowError:
        ppl = float("inf")
    return round(ppl, 3)


def exact_match_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int = 0,
) -> float:
    """
    Dizi seviyesinde tam eşleşme doğruluğunu yüzde olarak hesaplar.
    """
    batch_size = predictions.size(0)
    exact_matches = 0

    for i in range(batch_size):
        pred_seq = [t for t in predictions[i].tolist() if t != pad_idx]
        target_seq = [t for t in targets[i].tolist() if t != pad_idx]
        if pred_seq == target_seq:
            exact_matches += 1

    return round((exact_matches / batch_size) * 100.0, 2)


def token_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int = 0,
) -> float:
    """
    Dolgu (PAD) olmayan tokenlar üzerindeki doğru tahmin oranını yüzde olarak hesaplar.
    """
    mask = targets != pad_idx
    if mask.sum().item() == 0:
        return 0.0
    correct = ((predictions == targets) & mask).sum().item()
    total = mask.sum().item()
    return round((correct / total) * 100.0, 2)
