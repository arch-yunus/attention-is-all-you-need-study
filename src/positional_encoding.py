"""
Sinüzoidal Pozisyonel Kodlama (Positional Encoding) Modülü.

"Attention Is All You Need" (Vaswani et al., 2017) Bölüm 3.5:
PE_(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE_(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Analitik Sinüzoidal Pozisyonel Kodlama Katmanı.

    Model yineleme (recurrence) veya evrişim (convolution) içermediğinden,
    sıralı dizilim bilgisini vektör temsillerine enjekte etmek için sinüs ve
    kosinüs dalga boylarını kullanır.

    Args:
        d_model (int): Modelin gömme boyutu (Embedding dimension).
        dropout (float): Gömme ve pozisyon toplamına uygulanacak dropout oranı. Default: 0.1.
        max_len (int): Desteklenen azami dizi uzunluğu. Default: 5000.
    """

    def __init__(self, d_model: int = 512, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None

        # [max_len, d_model] boyutunda pozisyonel kodlama matrisi oluşturma
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]

        # Logaritma uzayında payda hesabı: 10000^(2i / d_model) = exp(2i * -log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        # Çift indekslere sinüs, tek indekslere kosinüs yerleştirme
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # [1, max_len, d_model] şeklinde genişletilerek batch boyutuna uyumlu hale getirilir
        pe = pe.unsqueeze(0)

        # register_buffer: Modelin state_dict'ine kaydedilir, GPU'ya taşınır ancak gradyan almaz
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Token gömmelerine pozisyon bilgisini ekler.

        Args:
            x (torch.Tensor): Gömme tensörü, şekil: [Batch, Seq_Len, d_model]

        Returns:
            torch.Tensor: Pozisyon bilgisi eklenmiş tensör, şekil: [Batch, Seq_Len, d_model]
        """
        # x.size(1) kadar pozisyon dilimi alınır ve x'e eklenir
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
