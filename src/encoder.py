"""
Kodlayıcı (Encoder) Modülü.

"Attention Is All You Need" (Vaswani et al., 2017) Bölüm 3.1:
Encoder, N = 6 adet özdeş EncoderLayer katmanının ardışıl yığılmasından oluşur.
"""

from typing import Optional
import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward
from .residual_norm import SublayerConnection, LayerNorm


class EncoderLayer(nn.Module):
    """
    Tekil Kodlayıcı Katmanı (Encoder Layer).

    İki ana alt katmandan oluşur:
    1. Çok Başlı Öz-Dikkat (Multi-Head Self-Attention)
    2. Konum Bazlı İleri Beslemeli Ağ (Position-wise Feed-Forward Network)
    Her iki alt katman da artık bağlantı (residual connection) ve LayerNorm ile sarılıdır.

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
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.sublayer_1 = SublayerConnection(size=d_model, dropout=dropout, norm_type=norm_type)
        self.sublayer_2 = SublayerConnection(size=d_model, dropout=dropout, norm_type=norm_type)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        İleri besleme.

        Args:
            x (torch.Tensor): Girdi tensörü, [Batch, Seq_Len, d_model]
            mask (Optional[torch.Tensor]): Kaynak padding maskesi, [Batch, 1, 1, Seq_Len]
        """
        # 1. Alt Katman: Self-Attention
        x = self.sublayer_1(x, lambda _x: self.self_attn(_x, _x, _x, mask=mask)[0])
        # 2. Alt Katman: Feed-Forward
        x = self.sublayer_2(x, self.feed_forward)
        return x


class Encoder(nn.Module):
    """
    N Katmanlı Kodlayıcı Yığını (Encoder Stack).

    Args:
        layer (EncoderLayer): Katman şablonu veya katman örneği.
        num_layers (int): Yığılacak katman sayısı (N = 6).
        d_model (int): Model boyutu.
        norm_type (str): 'post_ln' veya 'pre_ln'. Pre-LN durumunda çıkışta LayerNorm zorunludur.
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
                EncoderLayer(
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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Tüm katmanları sırayla çalıştırır.

        Args:
            x (torch.Tensor): Pozisyon eklenmiş token gömmeleri, [Batch, Seq_Len, d_model]
            mask (Optional[torch.Tensor]): Kaynak maskesi, [Batch, 1, 1, Seq_Len]
        """
        for layer in self.layers:
            x = layer(x, mask=mask)

        if self.norm is not None:
            x = self.norm(x)
        return x
