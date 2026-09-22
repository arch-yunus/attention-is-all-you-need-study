"""
Modern Dikkat Mimarileri ve Varyantları (Modern Attention Variants).

Bu modül, 2017 Transformer makalesinden günümüze (LLaMA, Mistral, PaLM, Gemma, Bloom)
dikkat mekanizmasının geçirdiği evrimi ve modern varyantlarını sunar:
1. Multi-Query Attention (MQA - Shazeer, 2019)
2. Grouped-Query Attention (GQA - Ainslie et al., 2023)
3. Rotary Position Embedding (RoPE - Su et al., 2021)
4. ALiBi Attention (Attention with Linear Biases - Press et al., 2021)
5. Sliding Window Attention (Mistral 7B / Longformer - Beltagy et al., 2020)
6. FlashAttention / PyTorch SDPA Wrapper (Dao et al., 2022)
"""

import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA - Noam Shazeer, 2019).
    
    Tüm sorgu başları (Query Heads) tek bir ortak Anahtar (Key) ve Değer (Value) başını paylaşır.
    KV-Cache bellek boyutunu H kat azaltarak çıkarım verimliliğini devasa oranda artırır.

    Args:
        d_model (int): Gizli katman boyutu.
        num_heads (int): Sorgu başı sayısı (H).
        dropout (float): Dropout oranı.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model num_heads'e tam bölünmelidir!"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Sorgu için H adet baş projeksiyonu
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        # Anahtar ve Değer için yalnızca TEK bir baş projeksiyonu (d_k)
        self.w_k = nn.Linear(d_model, self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, self.d_k, bias=False)

        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)

        # Q: [B, H, Seq_Q, d_k]
        q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        # K, V: [B, 1, Seq_K, d_k] -> broadcast edilir
        k = self.w_k(key).view(batch_size, seq_len_k, 1, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, seq_len_k, 1, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention: [B, H, Seq_Q, Seq_K]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        # Çıktı: [B, H, Seq_Q, d_k]
        context = torch.matmul(attn_weights, v)
        # Birleştirme: [B, Seq_Q, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = self.w_o(context)

        return output, attn_weights


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA - Ainslie et al., 2023).
    
    MHA ile MQA arasındaki altın dengedir (LLaMA-2/3 70B mimarisinde kullanılır).
    Sorgu başları G adet gruba ayrılır ve her grup kendi Key/Value başını paylaşır.

    Args:
        d_model (int): Gizli katman boyutu.
        num_heads (int): Toplam sorgu başı sayısı (H).
        num_kv_heads (int): Anahtar/Değer başı sayısı (G). num_heads % num_kv_heads == 0 olmalıdır.
        dropout (float): Dropout oranı.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model num_heads'e tam bölünmelidir!"
        assert num_heads % num_kv_heads == 0, "num_heads num_kv_heads'e tam bölünmelidir!"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)

        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)

        # Q: [B, H, Seq_Q, d_k]
        q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)

        # K, V: [B, G, Seq_K, d_k]
        k = self.w_k(key).view(batch_size, seq_len_k, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, seq_len_k, self.num_kv_heads, self.d_k).transpose(1, 2)

        # K ve V'yi H baş sayısına genişlet (repeat_interleave): [B, H, Seq_K, d_k]
        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = self.w_o(context)

        return output, attn_weights


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Rotary Position Embedding (RoPE) 2D rotasyon dönüşümü:
    x_rot = [x_1 * cos - x_2 * sin, x_1 * sin + x_2 * cos]
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE - Su et al., 2021).
    
    Konum bilgisini mutlak sinüs toplamı yerine sorgu ve anahtar vektörlerinin
    2B düzlemlerdeki açısal rotasyonu (dönmesi) olarak uygular.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Açı frekansları: theta_i = 1 / base^(2i / dim)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, self.inv_freq)  # [max_seq_len, dim // 2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_seq_len, dim]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q ve k tensörlerine rotasyon uygular.
        q, k: [Batch, num_heads, seq_len, d_k]
        offset: KV-Cache çıkarımında mevcut token konumu.
        """
        seq_len_curr = q.shape[2] if seq_len is None else seq_len
        cos = self.cos_cached[:, :, offset : offset + seq_len_curr, :]
        sin = self.sin_cached[:, :, offset : offset + seq_len_curr, :]
        return apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)


def get_alibi_slopes(num_heads: int) -> List[float]:
    """
    ALiBi (Press et al., 2021) baş eğim katsayılarını (m) hesaplar.
    m_h = 2^(-8h / H) geometrik dizisi ile her başa farklı eğim atanır.
    """
    def get_slopes_power_of_2(n: int) -> List[float]:
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        return get_slopes_power_of_2(num_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2)
        extra = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][: num_heads - closest_power_of_2]
        return slopes + extra


class ALiBiAttention(nn.Module):
    """
    Attention with Linear Biases (ALiBi - Press et al., 2021).
    
    Pozisyonel gömmeleri tamamen kaldırıp, dikkat matrisine token uzaklıkları ile
    orantılı sabit negatif eğim ekler:
    
    Attention_Score(i, j) = (q_i * k_j^T) / sqrt(d_k) - m_h * |i - j|
    
    Özellikleri:
    - Eğitimde görülen uzunluktan (örn. 2048) çok daha uzun dizilere (örn. 8192+)
      ekstrapolasyon yeteneği sağlar.
    - BLOOM ve MPT modellerinde tercih edilmiştir.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        slopes = torch.tensor(get_alibi_slopes(num_heads), dtype=torch.float32)
        self.register_buffer("slopes", slopes.view(1, num_heads, 1, 1), persistent=False)

    def _build_alibi_bias(self, seq_len_q: int, seq_len_k: int, device: torch.device) -> torch.Tensor:
        # [seq_len_q, 1] - [1, seq_len_k] -> mesafe matrisi: i - j
        pos_q = torch.arange(seq_len_q, device=device).unsqueeze(1)
        pos_k = torch.arange(seq_len_k, device=device).unsqueeze(0)
        relative_distance = pos_k - pos_q  # Nedensel durumda j <= i -> <= 0
        # ALiBi bias = slopes * relative_distance
        return self.slopes.to(device) * relative_distance.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_k, _ = key.shape

        q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        alibi_bias = self._build_alibi_bias(seq_len_q, seq_len_k, query.device)
        scores = scores + alibi_bias

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        return self.w_o(context), attn_weights


class SlidingWindowAttention(nn.Module):
    """
    Kayan Pencere Dikkati (Sliding Window Attention / SWA - Mistral 7B).
    
    Her token'ın sadece kendisinden önceki W adet token'a (yerel pencere) dikkat etmesini sağlar.
    Hesaplama karmaşıklığı O(N^2) yerine O(N * W)'ye düşerken, katmanlar üst üste bindikçe
    etkili alıcı alan (receptive field) L * W token'a ulaşır.
    
    Args:
        d_model (int): Model gizli boyutu.
        num_heads (int): Baş sayısı.
        window_size (int): Kayan pencere genişliği (W).
        dropout (float): Dropout oranı.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        window_size: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def _create_sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Nedensel mask: i >= j
        # Pencere mask: i - j < window_size
        i = torch.arange(seq_len, device=device).unsqueeze(1)
        j = torch.arange(seq_len, device=device).unsqueeze(0)
        mask = (i >= j) & (i - j < self.window_size)
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        window_mask = self._create_sliding_window_mask(seq_len, x.device)
        scores = scores.masked_fill(~window_mask, -1e9)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(context), attn_weights


class FlashAttentionSDPA(nn.Module):
    """
    PyTorch 2.0+ SDPA (Scaled Dot-Product Attention) Hızlandırıcı Katmanı.
    
    Eğer donanım ve PyTorch ortamı destekliyorsa FlashAttention v2 veya Bellek Verimli
    (Memory-Efficient) C++ / CUDA çekirdeklerini otomatik olarak kullanır.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_p = dropout

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_k, _ = key.shape

        q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)

        # PyTorch yerel SDPA çekirdeği (FlashAttention / MemEfficient / Math fallback)
        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal and (attn_mask is None),
        )

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        return self.w_o(out)


def compare_attention_memory(
    batch_size: int = 1,
    seq_len: int = 2048,
    num_heads: int = 32,
    d_model: int = 4096,
    num_layers: int = 32,
    precision_bytes: int = 2,  # FP16 / BF16 = 2 bytes
) -> dict:
    """
    MHA, GQA (8 grup) ve MQA için KV-Cache bellek boyutunu (Megabytes cinsinden) hesaplar.
    """
    d_k = d_model // num_heads

    def get_mb(kv_heads: int) -> float:
        total_bytes = batch_size * seq_len * 2 * kv_heads * d_k * num_layers * precision_bytes
        return total_bytes / (1024 * 1024)

    mha_mb = get_mb(num_heads)
    gqa_mb = get_mb(8)  # 8 KV heads
    mqa_mb = get_mb(1)  # 1 KV head

    return {
        "MHA_MB": round(mha_mb, 2),
        "GQA_MB": round(gqa_mb, 2),
        "MQA_MB": round(mqa_mb, 2),
        "GQA_Savings_Pct": round((1 - gqa_mb / mha_mb) * 100, 1),
        "MQA_Savings_Pct": round((1 - mqa_mb / mha_mb) * 100, 1),
    }
