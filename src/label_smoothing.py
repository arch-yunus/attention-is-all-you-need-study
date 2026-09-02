"""
Etiket Yumuşatma Kayıp Fonksiyonu (Label Smoothing Loss).

"Attention Is All You Need" (Vaswani et al., 2017) Bölüm 5.4:
Modelin hedeflere %100 kesinlikle kilitlenmesini engelleyerek aşırı güveni (overconfidence)
ve ezberlemeyi (overfitting) önleyen regülarizasyon tekniği.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """
    Etiket Yumuşatma ile Çapraz Entropi Kaybı (Label Smoothing Cross Entropy).

    Args:
        vocab_size (int): Kelime haznesi boyutu.
        padding_idx (int): Hesaba katılmayacak dolgu token indeksi. Varsayılan: 0.
        smoothing (float): Yumuşatma katsayısı (epsilon_ls). Orijinal makalede: 0.1.
    """

    def __init__(self, vocab_size: int, padding_idx: int = 0, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Kayıp hesaplama adımı.

        Args:
            logits (torch.Tensor): Model çıktı logitleri, [Batch * Seq_Len, vocab_size] veya [Batch, Seq_Len, vocab_size]
            target (torch.Tensor): Gerçek token indeksleri, [Batch * Seq_Len] veya [Batch, Seq_Len]

        Returns:
            torch.Tensor: Ortalama skalar kayıp değeri.
        """
        if logits.dim() > 2:
            logits = logits.contiguous().view(-1, logits.size(-1))
            target = target.contiguous().view(-1)

        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            # Dağıtılacak arka plan olasılığı: smoothing / (vocab_size - 2) (hedef ve pad hariç)
            smooth_val = self.smoothing / (self.vocab_size - 2) if self.vocab_size > 2 else self.smoothing
            true_dist.fill_(smooth_val)
            # Doğru hedefe güven olasılığı
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            # Dolgu indeksini sıfırla
            true_dist[:, self.padding_idx] = 0
            mask = target == self.padding_idx
            if mask.any():
                true_dist.masked_fill_(mask.unsqueeze(1), 0.0)

        loss = torch.sum(-true_dist * log_probs, dim=-1)
        non_pad_count = (target != self.padding_idx).sum().float()
        return loss.sum() / torch.clamp(non_pad_count, min=1.0)
