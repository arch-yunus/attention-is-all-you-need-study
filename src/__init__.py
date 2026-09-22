"""
Attention Is All You Need - Kapsamlı Referans ve Modern LLM Araştırma Kütüphanesi.
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
from .modern_layers import RMSNorm, SwiGLU, GeGLU, DeepNormScale
from .attention_variants import (
    MultiQueryAttention,
    GroupedQueryAttention,
    RotaryPositionalEmbedding,
    apply_rotary_emb,
    get_alibi_slopes,
    ALiBiAttention,
    SlidingWindowAttention,
    FlashAttentionSDPA,
    compare_attention_memory,
)
from .decoder_only import DecoderOnlyBlock, DecoderOnlyTransformer
from .tokenizer import BPETokenizer
from .generation import (
    greedy_decode,
    beam_search_decode,
    sample_decode,
    apply_sampling_filters,
)
from .metrics import (
    compute_bleu,
    corpus_bleu,
    calculate_perplexity,
    exact_match_accuracy,
    token_accuracy,
)
from .trainer import Trainer, TrainerConfig, TrainingHistory
from .visualizer import generate_html_dashboard

__all__ = [
    # Orijinal Mimari ve Katmanlar (Vaswani et al., 2017)
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
    # Modern Katman ve Aktivasyonlar (RMSNorm, SwiGLU, GeGLU, DeepNorm)
    "RMSNorm",
    "SwiGLU",
    "GeGLU",
    "DeepNormScale",
    # İleri Seviye Dikkat Varyantları (MQA, GQA, RoPE, ALiBi, SWA, FlashSDPA)
    "MultiQueryAttention",
    "GroupedQueryAttention",
    "RotaryPositionalEmbedding",
    "apply_rotary_emb",
    "get_alibi_slopes",
    "ALiBiAttention",
    "SlidingWindowAttention",
    "FlashAttentionSDPA",
    "compare_attention_memory",
    # Modern Salt-Dekoder (Decoder-Only) LLM Mimarisi
    "DecoderOnlyBlock",
    "DecoderOnlyTransformer",
    # BPE Alt Kelime Tokenizer'ı
    "BPETokenizer",
    # KV-Cache & Çıkarım / Örnekleme Algoritmaları
    "TransformerKVCache",
    "LayerKVCache",
    "greedy_decode",
    "beam_search_decode",
    "sample_decode",
    "apply_sampling_filters",
    # Değerlendirme Metrikleri
    "compute_bleu",
    "corpus_bleu",
    "calculate_perplexity",
    "exact_match_accuracy",
    "token_accuracy",
    # Eğitici ve Görselleştirici
    "Trainer",
    "TrainerConfig",
    "TrainingHistory",
    "generate_html_dashboard",
]
