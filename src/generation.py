"""
Otoregresif Üretim ve Kod Çözme Stratejileri (Generation & Decoding Strategies).

Bu modül, eğitilmiş Transformer ve Decoder-Only modelleri için modern çıkarım algoritmalarını sunar:
1. Açgözlü Çıkarım (Greedy Decoding) - KV-Cache destekli
2. Işın Araması (Beam Search) - Uzunluk ve tekrar cezası ile
3. Olasılıksal Örnekleme (Temperature, Top-k, Top-p / Nucleus, Min-p, Repetition & Frequency Penalty)
4. Evrensel Otoregresif Üretici (generate_autoregressive)
"""

import math
from typing import Optional, List, Union, Tuple, Any
import torch
import torch.nn.functional as F
from .transformer import Transformer
from .masks import generate_square_subsequent_mask


def apply_sampling_filters(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    prev_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Logit tensörüne Sıcaklık (Temperature), Tekrar Cezası (Repetition Penalty),
    Frekans Cezası (Frequency Penalty), Top-K, Min-P ve Top-P filtrelerini uygular.

    Args:
        logits (torch.Tensor): Model çıkış logitleri, [Batch, Vocab_Size]
        temperature (float): Sıcaklık değeri (T > 0).
        top_k (int): Yalnızca en yüksek olasılıklı K tokenı tut (0 = kapalı).
        top_p (float): Kümülatif olasılığı p eşiğini aşan tokenları tut (1.0 = kapalı).
        min_p (float): En olası tokenın olasılığının min_p katından küçük tokenları ele (0.0 = kapalı).
        repetition_penalty (float): Tekrar cezası katsayısı (1.0 = cezasız).
        frequency_penalty (float): Tekrarlanma sıklığına göre logit düşürme (0.0 = kapalı).
        prev_tokens (Optional[torch.Tensor]): Önceki üretilen tokenlar, [Batch, Seq_Len].

    Returns:
        torch.Tensor: Filtrelenmiş logit tensörü.
    """
    batch_size = logits.size(0)

    # 1. Tekrar Cezası (Repetition Penalty - Keskar et al., 2019)
    if repetition_penalty != 1.0 and prev_tokens is not None:
        for b in range(batch_size):
            for token_id in set(prev_tokens[b].tolist()):
                if logits[b, token_id] < 0:
                    logits[b, token_id] *= repetition_penalty
                else:
                    logits[b, token_id] /= repetition_penalty

    # 2. Frekans Cezası (Frequency Penalty)
    if frequency_penalty > 0.0 and prev_tokens is not None:
        for b in range(batch_size):
            token_counts = torch.bincount(prev_tokens[b], minlength=logits.size(-1))
            logits[b] -= frequency_penalty * token_counts.float()

    # 3. Sıcaklık Ölçekleme (Temperature Scaling)
    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature

    # 4. Top-K Filtreleme (Fan et al., 2018)
    if top_k > 0:
        top_k = min(max(top_k, 1), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # 5. Min-P Filtreleme (Grimminger et al., 2024)
    if 0.0 < min_p < 1.0:
        probs = F.softmax(logits, dim=-1)
        max_probs, _ = torch.max(probs, dim=-1, keepdim=True)
        min_p_threshold = max_probs * min_p
        indices_to_remove = probs < min_p_threshold
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # 6. Top-P (Nucleus) Filtreleme (Holtzman et al., 2019)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    return logits


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    max_len: int = 32,
    bos_idx: int = 1,
    eos_idx: int = 2,
    pad_idx: int = 0,
    use_cache: bool = True,
) -> torch.Tensor:
    """
    Açgözlü Kod Çözme (Greedy Decoding).
    Her adımda en yüksek logite sahip token'ı seçer.
    """
    model.eval()
    batch_size = src.size(0)
    device = src.device

    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    memory = model.encode(src, src_mask=src_mask)

    ys = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)

    if use_cache:
        kv_cache = model.create_kv_cache()
        for step in range(max_len - 1):
            current_input = ys[:, -1:]
            out = model.decode(
                current_input,
                memory=memory,
                src_mask=src_mask,
                tgt_mask=None,
                kv_cache=kv_cache,
                step=step,
            )
            logits = model.generator(out[:, -1])
            next_word = torch.argmax(logits, dim=-1, keepdim=True)
            ys = torch.cat([ys, next_word], dim=1)

            if (ys == eos_idx).any(dim=1).all():
                break
    else:
        for _ in range(max_len - 1):
            tgt_mask = generate_square_subsequent_mask(ys.size(1), device=device).unsqueeze(1)
            out = model.decode(ys, memory=memory, src_mask=src_mask, tgt_mask=tgt_mask)
            logits = model.generator(out[:, -1])
            next_word = torch.argmax(logits, dim=-1, keepdim=True)
            ys = torch.cat([ys, next_word], dim=1)

            if (ys == eos_idx).any(dim=1).all():
                break

    return ys


@torch.no_grad()
def sample_decode(
    model: Transformer,
    src: torch.Tensor,
    max_len: int = 32,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    min_p: float = 0.0,
    repetition_penalty: float = 1.1,
    frequency_penalty: float = 0.0,
    bos_idx: int = 1,
    eos_idx: int = 2,
    pad_idx: int = 0,
    use_cache: bool = True,
) -> torch.Tensor:
    """
    Olasılıksal Örnekleme ile Kod Çözme (Temperature, Top-k, Top-p, Min-p Sampling).
    """
    model.eval()
    batch_size = src.size(0)
    device = src.device

    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    memory = model.encode(src, src_mask=src_mask)

    ys = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)

    if use_cache:
        kv_cache = model.create_kv_cache()
        for step in range(max_len - 1):
            current_input = ys[:, -1:]
            out = model.decode(
                current_input,
                memory=memory,
                src_mask=src_mask,
                tgt_mask=None,
                kv_cache=kv_cache,
                step=step,
            )
            logits = model.generator(out[:, -1])

            filtered_logits = apply_sampling_filters(
                logits=logits.clone(),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                prev_tokens=ys,
            )
            probs = F.softmax(filtered_logits, dim=-1)
            next_word = torch.multinomial(probs, num_samples=1)
            ys = torch.cat([ys, next_word], dim=1)

            if (ys == eos_idx).any(dim=1).all():
                break
    else:
        for _ in range(max_len - 1):
            tgt_mask = generate_square_subsequent_mask(ys.size(1), device=device).unsqueeze(1)
            out = model.decode(ys, memory=memory, src_mask=src_mask, tgt_mask=tgt_mask)
            logits = model.generator(out[:, -1])

            filtered_logits = apply_sampling_filters(
                logits=logits.clone(),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                prev_tokens=ys,
            )
            probs = F.softmax(filtered_logits, dim=-1)
            next_word = torch.multinomial(probs, num_samples=1)
            ys = torch.cat([ys, next_word], dim=1)

            if (ys == eos_idx).any(dim=1).all():
                break

    return ys


@torch.no_grad()
def beam_search_decode(
    model: Transformer,
    src: torch.Tensor,
    beam_size: int = 4,
    max_len: int = 32,
    bos_idx: int = 1,
    eos_idx: int = 2,
    pad_idx: int = 0,
    length_penalty_alpha: float = 0.6,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """
    Işın Araması (Beam Search Decoding).
    """
    model.eval()
    device = src.device

    if src.size(0) > 1:
        results = []
        for i in range(src.size(0)):
            res = beam_search_decode(
                model=model,
                src=src[i : i + 1],
                beam_size=beam_size,
                max_len=max_len,
                bos_idx=bos_idx,
                eos_idx=eos_idx,
                pad_idx=pad_idx,
                length_penalty_alpha=length_penalty_alpha,
                repetition_penalty=repetition_penalty,
            )
            results.append(res)
        max_res_len = max(r.size(1) for r in results)
        padded_res = []
        for r in results:
            if r.size(1) < max_res_len:
                pad_tensor = torch.full(
                    (1, max_res_len - r.size(1)), pad_idx, dtype=torch.long, device=device
                )
                padded_res.append(torch.cat([r, pad_tensor], dim=1))
            else:
                padded_res.append(r)
        return torch.cat(padded_res, dim=0)

    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    memory = model.encode(src, src_mask=src_mask)

    initial_tokens = torch.tensor([[bos_idx]], dtype=torch.long, device=device)
    beams: List[Tuple[torch.Tensor, float, bool]] = [(initial_tokens, 0.0, False)]
    completed_beams: List[Tuple[torch.Tensor, float]] = []

    for step in range(max_len - 1):
        all_candidates: List[Tuple[torch.Tensor, float, bool]] = []

        for seq, score, finished in beams:
            if finished:
                all_candidates.append((seq, score, True))
                continue

            tgt_mask = generate_square_subsequent_mask(seq.size(1), device=device).unsqueeze(1)
            out = model.decode(seq, memory=memory, src_mask=src_mask, tgt_mask=tgt_mask)
            logits = model.generator(out[:, -1])

            if repetition_penalty != 1.0:
                for token_id in set(seq[0].tolist()):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty

            log_probs = F.log_softmax(logits, dim=-1)
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_size, dim=-1)

            for k in range(beam_size):
                next_token = topk_indices[0, k].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([seq, next_token], dim=1)
                new_score = score + topk_log_probs[0, k].item()
                is_eos = bool(next_token.item() == eos_idx)

                all_candidates.append((new_seq, new_score, is_eos))

        def get_normalized_score(cand: Tuple[torch.Tensor, float, bool]) -> float:
            seq_t, raw_score, _ = cand
            length = seq_t.size(1)
            lp = ((5.0 + length) / 6.0) ** length_penalty_alpha
            return raw_score / lp

        sorted_candidates = sorted(all_candidates, key=get_normalized_score, reverse=True)
        beams = sorted_candidates[:beam_size]

        for cand in beams:
            if cand[2]:
                completed_beams.append((cand[0], cand[1]))

        if all(cand[2] for cand in beams):
            break

    all_final = completed_beams if len(completed_beams) > 0 else [(b[0], b[1]) for b in beams]

    def get_final_score(cand: Tuple[torch.Tensor, float]) -> float:
        seq_t, raw_score = cand
        length = seq_t.size(1)
        lp = ((5.0 + length) / 6.0) ** length_penalty_alpha
        return raw_score / lp

    best_seq, _ = max(all_final, key=get_final_score)
    return best_seq
