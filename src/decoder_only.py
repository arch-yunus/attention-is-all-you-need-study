"""
Modern Salt-Dekoder (Decoder-Only) Dil Modeli Mimarisi (Causal Language Model).

Bu modül, 2017 Seq2Seq Transformer mimarisinden modern GPT, LLaMA-3, Mistral, Gemma
ve DeepSeek sınıfı büyük dil modellerine (LLM) geçişi sağlayan referans mimariyi sunar:
- Pre-LN yapısı (RMSNorm / LayerNorm)
- Döner Pozisyonel Gömmeler (RoPE) veya ALiBi
- Swish Kapılı Doğrusal Birimler (SwiGLU)
- Gruplanmış Sorgu Dikkati (GQA) veya Çok Başlı Dikkat (MHA)
- KV-Cache ile O(1) adımlı hızlı çıkarım
- Gelişmiş Token Üretimi (Greedy, Top-k, Top-p, Min-p, Repetition Penalty)
"""

import math
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modern_layers import RMSNorm, SwiGLU
from .residual_norm import LayerNorm
from .feed_forward import PositionwiseFeedForward
from .attention_variants import (
    GroupedQueryAttention,
    RotaryPositionalEmbedding,
    apply_rotary_emb,
)
from .kv_cache import LayerKVCache


class DecoderOnlyBlock(nn.Module):
    """
    Modern Decoder-Only Transformer Katman Bloğu (LLaMA / Mistral Tarzı).
    
    Akış:
    1. x_norm = Norm1(x)
    2. attn_out = Attention(x_norm)  [RoPE + GQA/MHA + Causal Mask]
    3. x = x + attn_out
    4. ffn_out = FFN(Norm2(x))       [SwiGLU / ReLU FFN]
    5. x = x + ffn_out
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        norm_type: str = "rmsnorm",
        ffn_type: str = "swiglu",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.d_k = d_model // num_heads

        # Projeksiyon matrisleri
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # Normalizasyon katmanları (Pre-LN)
        if norm_type == "rmsnorm":
            self.norm1 = RMSNorm(d_model, eps=eps)
            self.norm2 = RMSNorm(d_model, eps=eps)
        else:
            self.norm1 = LayerNorm(d_model, eps=eps)
            self.norm2 = LayerNorm(d_model, eps=eps)

        # İleri beslemeli ağ (FFN)
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, dropout=dropout, bias=False)
        else:
            self.ffn = PositionwiseFeedForward(
                d_model=d_model,
                d_ff=d_ff if d_ff is not None else 4 * d_model,
                dropout=dropout,
            )

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[RotaryPositionalEmbedding] = None,
        causal_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[LayerKVCache] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        İleri geçiş adımı.
        
        Args:
            x: [Batch, Seq_Len, d_model]
            rope: Döner pozisyon gömme modülü
            causal_mask: Üçgen nedensel maske [1, 1, Seq_Q, Seq_K]
            kv_cache: Bu katmana ait anahtar/değer önbelleği
            start_pos: Üretim sırasındaki mutlak token pozisyon ofseti
        """
        batch_size, seq_len, _ = x.shape
        h = self.norm1(x)

        # Q, K, V projeksiyonları
        q = self.w_q(h).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(h).view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.w_v(h).view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)

        # RoPE Rotasyonu
        if rope is not None:
            q, k = rope(q, k, seq_len=seq_len, offset=start_pos)

        # KV-Cache Güncellemesi (Çıkarım sırasında)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        # GQA için K ve V'yi H baş sayısına genişlet
        if self.num_queries_per_kv > 1:
            k_expanded = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v_expanded = v.repeat_interleave(self.num_queries_per_kv, dim=1)
        else:
            k_expanded = k
            v_expanded = v

        # Dikkat Skorları Hesabı
        scores = torch.matmul(q, k_expanded.transpose(-2, -1)) / math.sqrt(self.d_k)
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v_expanded)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_out = self.w_o(context)
        if self.dropout is not None:
            attn_out = self.dropout(attn_out)

        # 1. Artık Bağlantı
        x = x + attn_out

        # 2. Artık Bağlantı + FFN
        ffn_out = self.ffn(self.norm2(x))
        if self.dropout is not None:
            ffn_out = self.dropout(ffn_out)
        x = x + ffn_out

        return x, attn_weights


class DecoderOnlyTransformer(nn.Module):
    """
    Modern Decoder-Only Büyük Dil Modeli (Causal Language Model - LLaMA-3 / Mistral / GPT).
    
    Özellikler:
    - Pre-LN mimarisi + RMSNorm
    - Rotary Position Embedding (RoPE)
    - Grouped-Query Attention (GQA)
    - SwiGLU İleri Besleme Ağı
    - Ağırlık Bağlama (Weight Tying) opsiyonu
    - Uçtan uca CrossEntropyLoss hesaplama
    - KV-Cache ile hızlı autoregressive metin üretimi
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        d_ff: Optional[int] = None,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        norm_type: str = "rmsnorm",
        ffn_type: str = "swiglu",
        rope_base: float = 10000.0,
        share_weights: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.max_seq_len = max_seq_len

        # Token Embedding
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)

        # Döner Pozisyonel Kodlama (RoPE)
        self.rope = RotaryPositionalEmbedding(
            dim=d_model // num_heads,
            max_seq_len=max_seq_len,
            base=rope_base,
        )

        # Katman Blokları
        self.layers = nn.ModuleList([
            DecoderOnlyBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                d_ff=d_ff,
                dropout=dropout,
                norm_type=norm_type,
                ffn_type=ffn_type,
                eps=eps,
            )
            for _ in range(num_layers)
        ])

        # Final Normalizasyon
        if norm_type == "rmsnorm":
            self.final_norm = RMSNorm(d_model, eps=eps)
        else:
            self.final_norm = LayerNorm(d_model, eps=eps)

        # Dil Modeli Çıkış Başlığı (LM Head)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if share_weights:
            self.lm_head.weight = self.tok_embeddings.weight

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Alt üçgen maskesi: [1, 1, seq_len, seq_len]
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[LayerKVCache]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Model ileri geçişi.
        
        Args:
            tokens: [Batch, Seq_Len]
            targets: [Batch, Seq_Len] - Eğer verilirse CrossEntropyLoss hesaplanır.
            kv_caches: Her katman için LayerKVCache listesi
            start_pos: Mevcut token başlangıç pozisyonu
        
        Returns:
            logits: [Batch, Seq_Len, Vocab_Size]
            loss: Optional[torch.Tensor]
        """
        batch_size, seq_len = tokens.shape
        x = self.tok_embeddings(tokens)

        # Causal mask (Eğer tek token üretilmiyorsa)
        causal_mask = None
        if seq_len > 1:
            causal_mask = self._build_causal_mask(seq_len, tokens.device)

        for i, layer in enumerate(self.layers):
            cache_i = kv_caches[i] if kv_caches is not None else None
            x, _ = layer(
                x,
                rope=self.rope,
                causal_mask=causal_mask,
                kv_cache=cache_i,
                start_pos=start_pos,
            )

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Shifted cross-entropy: [B*S, Vocab] vs [B*S]
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        min_p: float = 0.05,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        KV-Cache ve gelişmiş filtreler (Top-k, Top-p, Min-p, Repetition Penalty) ile
        hızlı metin üretimi.
        
        Args:
            prompt_tokens: [Batch, Prompt_Len]
        Returns:
            generated_tokens: [Batch, Prompt_Len + Generated_Len]
        """
        self.eval()
        batch_size, prompt_len = prompt_tokens.shape
        device = prompt_tokens.device

        generated = prompt_tokens.clone()

        if use_cache:
            kv_caches = [LayerKVCache() for _ in range(self.num_layers)]
            # Prefill Aşaması (Tüm istemi tek seferde işle)
            logits, _ = self.forward(prompt_tokens, kv_caches=kv_caches, start_pos=0)
            next_token_logits = logits[:, -1, :]

            curr_pos = prompt_len
            for _ in range(max_new_tokens):
                # Repetition Penalty
                if repetition_penalty != 1.0:
                    for b in range(batch_size):
                        for prev_tok in set(generated[b].tolist()):
                            if next_token_logits[b, prev_tok] > 0:
                                next_token_logits[b, prev_tok] /= repetition_penalty
                            else:
                                next_token_logits[b, prev_tok] *= repetition_penalty

                # Sıcaklık ve Örnekleme Filtreleri
                if temperature == 0.0:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    scaled_logits = next_token_logits / temperature
                    filtered_logits = self._apply_sampling_filters(scaled_logits, top_k, top_p, min_p)
                    probs = F.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=1)

                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

                # Decoding Adımı (Yalnızca üretilen 1 tokeni modele ver, O(1) maliyet)
                logits, _ = self.forward(next_token, kv_caches=kv_caches, start_pos=curr_pos)
                next_token_logits = logits[:, -1, :]
                curr_pos += 1
        else:
            # Cache'siz Standart Adım
            for _ in range(max_new_tokens):
                logits, _ = self.forward(generated)
                next_token_logits = logits[:, -1, :]
                if temperature == 0.0:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    scaled_logits = next_token_logits / temperature
                    filtered_logits = self._apply_sampling_filters(scaled_logits, top_k, top_p, min_p)
                    probs = F.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

        return generated

    @staticmethod
    def _apply_sampling_filters(
        logits: torch.Tensor,
        top_k: int = 50,
        top_p: float = 0.9,
        min_p: float = 0.05,
    ) -> torch.Tensor:
        """
        Top-K, Top-P (Nucleus) ve Min-P filtrelerini ardışık olarak uygular.
        """
        filtered = logits.clone()

        # 1. Top-K Filtresi
        if top_k > 0:
            top_k_val = min(top_k, logits.size(-1))
            indices_to_remove = filtered < torch.topk(filtered, top_k_val)[0][..., -1, None]
            filtered[indices_to_remove] = -float("Inf")

        # 2. Min-P Filtresi: (p >= p_max * min_p)
        if 0.0 < min_p < 1.0:
            probs = F.softmax(filtered, dim=-1)
            max_probs, _ = torch.max(probs, dim=-1, keepdim=True)
            min_p_threshold = max_probs * min_p
            filtered[probs < min_p_threshold] = -float("Inf")

        # 3. Top-P (Nucleus) Filtresi
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            # En az bir tokeni koru
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            filtered[indices_to_remove] = -float("Inf")

        return filtered
