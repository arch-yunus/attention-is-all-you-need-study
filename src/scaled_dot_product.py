"""
Scaled Dot-Product Attention Modülü.

"Attention Is All You Need" (Vaswani et al., 2017) Bölüm 3.2.1:
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Ölçeklenmiş Nokta Çarpım Dikkat Mekanizması (Scaled Dot-Product Attention).

    Sorgu (Q), Anahtar (K) ve Değer (V) tensörleri arasındaki benzerliği hesaplar.
    İç çarpım değerleri sqrt(d_k) ile bölünerek varyans 1.0 seviyesinde tutulur
    ve softmax fonksiyonunun aşırı doyum (gradient saturation) bölgesine girmesi önlenir.

    Args:
        dropout (float): Softmax sonrasındaki dikkat ağırlıklarına uygulanacak dropout oranı. Default: 0.1
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        İleri besleme adımı.

        Args:
            query (torch.Tensor): Sorgu tensörü, şekil: [Batch, Heads, Seq_Len_Q, d_k]
            key (torch.Tensor): Anahtar tensörü, şekil: [Batch, Heads, Seq_Len_K, d_k]
            value (torch.Tensor): Değer tensörü, şekil: [Batch, Heads, Seq_Len_K, d_v]
            mask (Optional[torch.Tensor]): Maskeleme tensörü (1: izin verilen, 0: maskelenecek veya boolean),
                                          şekil: [Batch, 1, 1, Seq_Len_K] veya [Batch, 1, Seq_Len_Q, Seq_Len_K]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - output: Dikkat uygulanmış tensör, şekil: [Batch, Heads, Seq_Len_Q, d_v]
                - attention_weights: Softmax normalize dikkat haritası, şekil: [Batch, Heads, Seq_Len_Q, Seq_Len_K]
        """
        d_k = query.size(-1)

        # 1. Q ve K^T matris çarpımı: [B, H, S_q, d_k] x [B, H, d_k, S_k] -> [B, H, S_q, S_k]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # 2. Maske uygulama (Padding maskesi veya Causal/Look-ahead maskesi)
        if mask is not None:
            # Maske boolean ise False olan yerler, 0/1 ise 0 olan yerler maskelenir
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float("-1e9"))
            else:
                scores = scores.masked_fill(mask == 0, float("-1e9"))

        # 3. Softmax ile satır bazında olasılık dağılımı elde etme
        attention_weights = F.softmax(scores, dim=-1)

        # 4. Dropout (Eğitim sırasında aşırı uyumu önlemek için rastgele başları regüle eder)
        if self.dropout is not None:
            attention_weights = self.dropout(attention_weights)

        # 5. Ağırlıklar ile Değer (V) matris çarpımı: [B, H, S_q, S_k] x [B, H, S_k, d_v] -> [B, H, S_q, d_v]
        output = torch.matmul(attention_weights, value)

        return output, attention_weights
