"""
Anahtar-Değer Önbelleği (Key-Value Cache / KV-Cache) Modülü.

Otoregresif üretim (inference) sırasında önceki adımlarda hesaplanan Anahtar (Key)
ve Değer (Value) projeksiyonlarının tekrar hesaplanmasını önleyerek zaman karmaşıklığını
O(N^2) seviyesinden adım başı O(N) maliyetine indirir.
"""

from typing import Optional, Tuple, Dict, Any
import torch


class LayerKVCache:
    """
    Tekil bir dikkat katmanı için Anahtar-Değer önbelleği.
    Hem öz-dikkat (self-attention) hem de çapraz dikkat (cross-attention) durumlarını destekler.
    """

    def __init__(self) -> None:
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def update(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Yeni adımın Key ve Value tensörlerini mevcut önbelleğe ekler.

        Args:
            new_k (torch.Tensor): [Batch, num_heads, seq_len, d_k]
            new_v (torch.Tensor): [Batch, num_heads, seq_len, d_k]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Güncellenmiş birleşik (K, V) tensörleri.
        """
        if self.k is None or self.v is None:
            self.k = new_k
            self.v = new_v
        else:
            self.k = torch.cat([self.k, new_k], dim=2)
            self.v = torch.cat([self.v, new_v], dim=2)
        return self.k, self.v

    def reset(self) -> None:
        """Önbelleği sıfırlar."""
        self.k = None
        self.v = None

    @property
    def seq_len(self) -> int:
        """Mevcut önbellekteki dizi uzunluğu."""
        return 0 if self.k is None else self.k.size(2)


class TransformerKVCache:
    """
    Tüm Transformer Kod Çözücü katmanları için global KV-Cache yöneticisi.
    """

    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self.self_attn_caches = [LayerKVCache() for _ in range(num_layers)]
        self.cross_attn_caches = [LayerKVCache() for _ in range(num_layers)]

    def get_self_attn_cache(self, layer_idx: int) -> LayerKVCache:
        return self.self_attn_caches[layer_idx]

    def get_cross_attn_cache(self, layer_idx: int) -> LayerKVCache:
        return self.cross_attn_caches[layer_idx]

    def reset(self) -> None:
        """Tüm katmanların önbelleklerini temizler."""
        for cache in self.self_attn_caches:
            cache.reset()
        for cache in self.cross_attn_caches:
            cache.reset()

    @property
    def current_seq_len(self) -> int:
        """İlk katmanın self-attention önbellek uzunluğunu döner."""
        if not self.self_attn_caches:
            return 0
        return self.self_attn_caches[0].seq_len
