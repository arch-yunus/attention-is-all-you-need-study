"""
Modern Katman ve Aktivasyon Mimarileri (Modern Transformer Layers).

2017 Transformer mimarisinden günümüz modern LLM'lerine (LLaMA-3, Mistral, PaLM, Gemma, Chinchilla)
kadar geliştirilen temel mimari bloklar:
1. RMSNorm (Root Mean Square Layer Normalization - Zhang & Sennrich, 2019)
2. SwiGLU (Swish Gated Linear Unit - Shazeer, 2020)
3. GeGLU (GELU Gated Linear Unit - Shazeer, 2020)
4. DeepNorm Scaling (Wang et al., 2022 - DeepNet)
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Kök Ortalama Kare Normalizasyonu (Root Mean Square Normalization - RMSNorm).
    
    Zhang & Sennrich (2019) tarafından önerilmiş; LLaMA, Mistral ve Gemma gibi
    modern büyük dil modellerinde standart hale gelmiştir.
    
    Klasik LayerNorm'daki ortalama merkezleme (mean centering) işlemini kaldırarak:
    RMS(x) = sqrt( (1 / d) * sum(x_i^2) + eps )
    y = (x / RMS(x)) * gamma
    
    Avantajları:
    - %10 - %50 arası hesaplama ve bellek bant genişliği tasarrufu sağlar.
    - Ölçekleme değişmezliği (scale invariance) özelliğini korur.
    - Kararlı gradyan akışını destekler.
    
    Args:
        dim (int): Normalizasyon yapılacak özellik boyutu (d_model).
        eps (float): Sayısal kararlılık sabiti (Varsayılan: 1e-6).
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Giriş tensörü, [..., dim]
        Returns:
            torch.Tensor: RMS ile normalize edilmiş tensör, [..., dim]
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwiGLU(nn.Module):
    """
    Swish Kapılı Doğrusal Birim (Swish Gated Linear Unit - SwiGLU).
    
    Noam Shazeer (2020) "GLU Variants Improve Transformer" makalesinde önerilmiştir.
    LLaMA, Mistral, PaLM ve Gemma'nın varsayılan ileri besleme (FFN) katmanıdır.
    
    Matematiksel Formül:
    SwiGLU(x) = (Swish(x * W_gate) * (x * W_up)) * W_down
    Burada Swish(z) = z * sigmoid(z) (diğer adıyla SiLU).
    
    Parametre Eşitliği Notu:
    Standart FFN'de ara boyut 4 * d_model iken, SwiGLU 3 adet ağırlık matrisi
    içerdiğinden parametre sayısını dengede tutmak için genellikle:
    d_ff = int(2/3 * 4 * d_model) veya 8/3 * d_model olarak seçilir.
    
    Args:
        d_model (int): Model gizli boyutu.
        d_ff (Optional[int]): Ara katman boyutu. Belirtilmezse (8/3 * d_model) ve 256'nın katına yuvarlanır.
        dropout (float): Dropout oranı.
        bias (bool): Lineer katmanlarda bias kullanılıp kullanılmayacağı (Modern LLM'lerde False tercih edilir).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_ff is None:
            hidden_dim = int(2 * (4 * d_model) / 3)
            hidden_dim = 256 * ((hidden_dim + 256 - 1) // 256)
            d_ff = hidden_dim

        self.d_model = d_model
        self.d_ff = d_ff

        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Giriş tensörü, [Batch, Seq_Len, d_model]
        Returns:
            torch.Tensor: Çıkış tensörü, [Batch, Seq_Len, d_model]
        """
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        hidden = gate * up
        if self.dropout is not None:
            hidden = self.dropout(hidden)
        return self.w_down(hidden)


class GeGLU(nn.Module):
    """
    GELU Kapılı Doğrusal Birim (GELU Gated Linear Unit - GeGLU).
    
    T5 v1.1 ve çeşitli multimodal modellerde kullanılan kapılı FFN varyantı.
    
    Matematiksel Formül:
    GeGLU(x) = (GELU(x * W_gate) * (x * W_up)) * W_down
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_ff is None:
            hidden_dim = int(2 * (4 * d_model) / 3)
            hidden_dim = 256 * ((hidden_dim + 256 - 1) // 256)
            d_ff = hidden_dim

        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.w_gate(x))
        up = self.w_up(x)
        hidden = gate * up
        if self.dropout is not None:
            hidden = self.dropout(hidden)
        return self.w_down(hidden)


class DeepNormScale(nn.Module):
    """
    DeepNorm Ölçekleme Katmanı (Wang et al., 2022 - DeepNet).
    
    1000+ katmanlı ultra-derin Transformer modellerinin (DeepNet) Post-LN mimarisinde
    gradyan patlaması olmadan eğitilmesini sağlayan artık bağlantı ölçekleyicisidir:
    x_next = LayerNorm(x * alpha + Sublayer(x))
    """

    def __init__(self, num_encoder_layers: int, num_decoder_layers: int = 0) -> None:
        super().__init__()
        n = num_encoder_layers
        m = num_decoder_layers
        if m > 0:
            self.alpha = (2.0 * n)**0.25
            self.beta = (8.0 * n)**(-0.25)
        else:
            self.alpha = (2.0 * n)**0.25
            self.beta = (2.0 * n)**(-0.25)

    def forward(self, x: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        return x * self.alpha + sublayer_out
