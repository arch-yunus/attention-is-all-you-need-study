"""
Multi-Head Attention Modülü.

"Attention Is All You Need" (Vaswani et al., 2017) Bölüm 3.2.2:
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O
nerede head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from .scaled_dot_product import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Çok Başlı Dikkat Mekanizması (Multi-Head Attention).

    Modelin farklı konumlardaki farklı temsil alt uzaylarındaki (subspaces)
    bilgilere eş zamanlı olarak odaklanmasını sağlar.

    Args:
        d_model (int): Modelin gizli katman boyutu (Orijinal makalede: 512).
        num_heads (int): Paralel dikkat başı sayısı (Orijinal makalede: 8).
        dropout (float): Dikkat ağırlıklarına ve çıkış projeksiyonuna uygulanacak dropout. Default: 0.1.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) num_heads ({num_heads}) değerine tam bölünmelidir!"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Genellikle 512 // 8 = 64

        # Doğrusal Projeksiyon Katmanları: W^Q, W^K, W^V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Çekirdek Scaled Dot-Product Attention mekanizması
        self.attention = ScaledDotProductAttention(dropout=dropout)

        # Çıkış birleştirme projeksiyonu: W^O
        self.w_o = nn.Linear(d_model, d_model, bias=False)
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
            query (torch.Tensor): Sorgu tensörü, şekil: [Batch, Seq_Len_Q, d_model]
            key (torch.Tensor): Anahtar tensörü, şekil: [Batch, Seq_Len_K, d_model]
            value (torch.Tensor): Değer tensörü, şekil: [Batch, Seq_Len_K, d_model]
            mask (Optional[torch.Tensor]): Maskeleme tensörü

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - output: Dikkat çıktısı, şekil: [Batch, Seq_Len_Q, d_model]
                - attention_weights: Dikkat haritası, şekil: [Batch, num_heads, Seq_Len_Q, Seq_Len_K]
        """
        batch_size = query.size(0)

        # 1. Doğrusal Projeksiyonlar: [B, S, d_model] -> [B, S, d_model]
        # Ardından başlara ayrıştırma ve transpozisyon: [B, S, h, d_k] -> [B, h, S, d_k]
        q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. Scaled Dot-Product Attention uygulama
        # scores: [B, h, Seq_Len_Q, d_k], attn_weights: [B, h, Seq_Len_Q, Seq_Len_K]
        out, attention_weights = self.attention(q, k, v, mask=mask)

        # 3. Başları birleştirme (Concatenation):
        # [B, h, Seq_Len_Q, d_k] -> [B, Seq_Len_Q, h, d_k] -> [B, Seq_Len_Q, d_model]
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 4. Son doğrusal çıkış projeksiyonu (W^O) ve dropout
        output = self.w_o(out)
        if self.dropout is not None:
            output = self.dropout(output)

        return output, attention_weights
