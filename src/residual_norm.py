"""
Artık Bağlantı (Residual Connection) ve Katman Normalizasyonu (LayerNorm) Modülü.

"Attention Is All You Need" (Vaswani et al., 2017) Bölüm 3.1:
Output = LayerNorm(x + Sublayer(x))   [Post-LN: Orijinal Makale]
Modern Alternatif: Output = x + Sublayer(LayerNorm(x))   [Pre-LN]
"""

from typing import Callable
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Katman Normalizasyonu (Layer Normalization).

    Öznitelik boyutu (d_model) ekseni boyunca ortalamayı sıfır, varyansı bir yapar:
    y = ((x - E[x]) / sqrt(Var[x] + eps)) * gamma + beta

    Args:
        features (int): Normalizasyon yapılacak öznitelik boyutu (d_model).
        eps (float): Sıfıra bölme hatasını önleyen sayısal kararlılık sabiti. Varsayılan: 1e-6.
    """

    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SublayerConnection(nn.Module):
    """
    Alt Katman Bağlantısı (Residual Connection + Dropout + LayerNorm).

    Hem 2017 orijinal makalesindeki Post-LN mimarisini hem de günümüz LLM'lerinde
    gradyan kararlılığı sağlayan Pre-LN mimarisini destekler.

    Args:
        size (int): Model gizli boyutu (d_model).
        dropout (float): Alt katman çıktısına eklenecek dropout oranı.
        norm_type (str): 'post_ln' (orijinal makale) veya 'pre_ln' (modern alternatif).
    """

    def __init__(self, size: int, dropout: float = 0.1, norm_type: str = "post_ln") -> None:
        super().__init__()
        assert norm_type in ["post_ln", "pre_ln"], "norm_type 'post_ln' veya 'pre_ln' olmalıdır."
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None
        self.norm_type = norm_type

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Alt katmanı artık bağlantı ve normalizasyonla sarar.

        Args:
            x (torch.Tensor): Giriş tensörü.
            sublayer (Callable): Uygulanacak alt katman (örn. Attention veya FFN).
        """
        if self.norm_type == "post_ln":
            # Orijinal Vaswani vd. (2017) yaklaşımı:
            # Önce alt katman uygulanır, x ile toplanır, sonra normalize edilir:
            sub_out = sublayer(x)
            if self.dropout is not None:
                sub_out = self.dropout(sub_out)
            return self.norm(x + sub_out)
        else:
            # Modern Pre-LN yaklaşımı (GPT, LLaMA):
            # Önce normalize edilir, alt katmandan geçirilir, sonra doğrudan toplanır:
            sub_out = sublayer(self.norm(x))
            if self.dropout is not None:
                sub_out = self.dropout(sub_out)
            return x + sub_out
