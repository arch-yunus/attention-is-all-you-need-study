"""
Position-wise Feed-Forward Network Modülü.

"Attention Is All You Need" (Vaswani et al., 2017) Bölüm 3.3:
FFN(x) = max(0, x * W_1 + b_1) * W_2 + b_2
"""

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    Konum Bazlı İleri Beslemeli Ağ (Position-wise Feed-Forward Network).

    Her bir token konumuna bağımsız ve özdeş olarak uygulanan iki katmanlı
    doğrusal dönüşüm ve aralarındaki doğrusal olmayan (non-linear) aktivasyon fonksiyonudur.

    Args:
        d_model (int): Giriş ve çıkış vektör boyutu (Varsayılan: 512).
        d_ff (int): Ara gizli katman genişleme boyutu (Varsayılan: 2048).
        dropout (float): Ara katmanda uygulanacak dropout oranı. Varsayılan: 0.1.
    """

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        İleri besleme adımı.

        Args:
            x (torch.Tensor): Giriş tensörü, şekil: [Batch, Seq_Len, d_model]

        Returns:
            torch.Tensor: Çıkış tensörü, şekil: [Batch, Seq_Len, d_model]
        """
        # [Batch, Seq_Len, d_model] -> [Batch, Seq_Len, d_ff]
        hidden = self.relu(self.w_1(x))
        if self.dropout is not None:
            hidden = self.dropout(hidden)
        # [Batch, Seq_Len, d_ff] -> [Batch, Seq_Len, d_model]
        return self.w_2(hidden)
