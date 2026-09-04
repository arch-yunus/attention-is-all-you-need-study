"""
Attention Is All You Need - Kapsamlı Referans ve Araştırma Kütüphanesi.
"""

from .scaled_dot_product import ScaledDotProductAttention
from .multi_head_attention import MultiHeadAttention
from .positional_encoding import PositionalEncoding
from .feed_forward import PositionwiseFeedForward
from .residual_norm import LayerNorm, SublayerConnection
from .encoder import EncoderLayer, Encoder
from .decoder import DecoderLayer, Decoder
from .transformer import Transformer, Generator
from .masks import generate_square_subsequent_mask, create_padding_mask, create_masks
from .optimizer import NoamLR
from .label_smoothing import LabelSmoothingLoss
from .kv_cache import TransformerKVCache, LayerKVCache
from .generation import (
    greedy_decode,
    beam_search_decode,
    sample_decode,
    apply_sampling_filters,
)
from .attention_variants import (
    MultiQueryAttention,
    GroupedQueryAttention,
    RotaryPositionalEmbedding,
    apply_rotary_emb,
    compare_attention_memory,
)
from .metrics import (
    compute_bleu,
    corpus_bleu,
    calculate_perplexity,
    exact_match_accuracy,
    token_accuracy,
)
from .trainer import Trainer, TrainerConfig, TrainingHistory

__all__ = [
    # Orijinal Mimari ve Katmanlar
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "PositionalEncoding",
    "PositionwiseFeedForward",
    "LayerNorm",
    "SublayerConnection",
    "EncoderLayer",
    "Encoder",
    "DecoderLayer",
    "Decoder",
    "Transformer",
    "Generator",
    "generate_square_subsequent_mask",
    "create_padding_mask",
    "create_masks",
    "NoamLR",
    "LabelSmoothingLoss",
    # KV-Cache & Çıkarım Algoritmaları
    "TransformerKVCache",
    "LayerKVCache",
    "greedy_decode",
    "beam_search_decode",
    "sample_decode",
    "apply_sampling_filters",
    # Modern Dikkat Varyantları (LLaMA / Mistral / PaLM)
    "MultiQueryAttention",
    "GroupedQueryAttention",
    "RotaryPositionalEmbedding",
    "apply_rotary_emb",
    "compare_attention_memory",
    # Değerlendirme Metrikleri
    "compute_bleu",
    "corpus_bleu",
    "calculate_perplexity",
    "exact_match_accuracy",
    "token_accuracy",
    # Eğitici ve Kontrol Noktaları
    "Trainer",
    "TrainerConfig",
    "TrainingHistory",
]
