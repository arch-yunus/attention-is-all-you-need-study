"""
Kod Çözücü (Decoder) Modülü.

"Attention Is All You Need" (Vaswani et al., 2017) Bölüm 3.1:
Decoder, N = 6 adet özdeş DecoderLayer katmanının ardışıl yığılmasından oluşur.
Her katman 3 alt katmana sahiptir:
1. Maskeli Çok Başlı Öz-Dikkat (Masked Self-Attention)
2. Çapraz Dikkat (Cross-Attention: Q decoder'dan, K ve V encoder'dan gelir)
3. Konum Bazlı İleri Beslemeli Ağ (Position-wise Feed-Forward)
"""

from typing import Optional
import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward
from .residual_norm import SublayerConnection, LayerNorm


class DecoderLayer(nn.Module):
    """
    Tekil Kod Çözücü Katmanı (Decoder Layer).

    Args:
        d_model (int): Model boyutu (512).
        num_heads (int): Dikkat başı sayısı (8).
        d_ff (int): FFN ara katman boyutu (2048).
        dropout (float): Dropout oranı (0.1).
        norm_type (str): 'post_ln' veya 'pre_ln'.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        norm_type: str = "post_ln",
    ) -> None:
        super().__init__()
        # 1. Maskelenmiş Öz-Dikkat
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        # 2. Çapraz Dikkat (Encoder-Decoder Dikkat Köprüsü)
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        # 3. İleri Beslemeli Ağ
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        # 3 alt katmanın artık bağlantıları
        self.sublayer_1 = SublayerConnection(size=d_model, dropout=dropout, norm_type=norm_type)
        self.sublayer_2 = SublayerConnection(size=d_model, dropout=dropout, norm_type=norm_type)
        self.sublayer_3 = SublayerConnection(size=d_model, dropout=dropout, norm_type=norm_type)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        İleri besleme.

        Args:
            x (torch.Tensor): Hedef tensör, [Batch, Seq_Len_Tgt, d_model]
            memory (torch.Tensor): Encoder çıktısı, [Batch, Seq_Len_Src, d_model]
            src_mask (Optional[torch.Tensor]): Kaynak maskesi (Padding), [Batch, 1, 1, Seq_Len_Src]
            tgt_mask (Optional[torch.Tensor]): Hedef maskesi (Causal + Padding), [Batch, 1, Seq_Len_Tgt, Seq_Len_Tgt]

        Returns:
            torch.Tensor: Decoder katman çıktısı, [Batch, Seq_Len_Tgt, d_model]
        """
        # 1. Alt Katman: Masked Self-Attention (Gelecekteki kelimeleri görmesi engellenir)
        x = self.sublayer_1(x, lambda _x: self.self_attn(_x, _x, _x, mask=tgt_mask)[0])

        # 2. Alt Katman: Cross-Attention (Q hedef diziden, K ve V kaynak diziden gelir)
        x = self.sublayer_2(x, lambda _x: self.cross_attn(_x, memory, memory, mask=src_mask)[0])

        # 3. Alt Katman: Feed-Forward
        x = self.sublayer_3(x, self.feed_forward)
        return x


class Decoder(nn.Module):
    """
    N Katmanlı Kod Çözücü Yığını (Decoder Stack).

    Args:
        num_layers (int): Yığılacak katman sayısı (N = 6).
        d_model (int): Model boyutu.
        norm_type (str): 'post_ln' veya 'pre_ln'.
    """

    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        norm_type: str = "post_ln",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNorm(d_model) if norm_type == "pre_ln" else None

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Tüm decoder katmanlarını sırayla çalıştırır.

        Args:
            x (torch.Tensor): Hedef tensör, [Batch, Seq_Len_Tgt, d_model]
            memory (torch.Tensor): Encoder çıktısı, [Batch, Seq_Len_Src, d_model]
            src_mask (Optional[torch.Tensor]): Kaynak maskesi, [Batch, 1, 1, Seq_Len_Src]
            tgt_mask (Optional[torch.Tensor]): Hedef maskesi, [Batch, 1, Seq_Len_Tgt, Seq_Len_Tgt]
        """
        for layer in self.layers:
            x = layer(x, memory=memory, src_mask=src_mask, tgt_mask=tgt_mask)

        if self.norm is not None:
            x = self.norm(x)
        return x
