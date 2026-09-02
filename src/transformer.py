"""
Transformer Modeli (Uçtan Uca Referans Mimarisi).

"Attention Is All You Need" (Vaswani et al., 2017) Bölüm 3:
Encoder-Decoder Mimarisi, Pozisyonel Kodlama ve Lineer Jeneratör.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .positional_encoding import PositionalEncoding
from .masks import create_masks


class Generator(nn.Module):
    """
    Standart Doğrusal Projeksiyon ve Log-Softmax Katmanı.
    Decoder çıkışını kelime haznesi (vocab) log-olasılıklarına dönüştürür.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Transformer(nn.Module):
    """
    Vaswani et al. (2017) "Attention Is All You Need" referans Transformer modeli.

    Args:
        src_vocab_size (int): Kaynak dil kelime haznesi boyutu.
        tgt_vocab_size (int): Hedef dil kelime haznesi boyutu.
        d_model (int): Modelin gizli boyutu (Varsayılan: 512).
        num_heads (int): Çoklu dikkat başı sayısı (Varsayılan: 8).
        num_encoder_layers (int): Encoder katman sayısı (Varsayılan: 6).
        num_decoder_layers (int): Decoder katman sayısı (Varsayılan: 6).
        d_ff (int): FFN ara katman boyutu (Varsayılan: 2048).
        dropout (float): Dropout oranı (Varsayılan: 0.1).
        max_len (int): Azami dizi uzunluğu (Varsayılan: 5000).
        norm_type (str): 'post_ln' (orijinal) veya 'pre_ln'.
        share_weights (bool): Hedef embedding ile jeneratör ağırlıklarını bağlama (Weight Tying).
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        norm_type: str = "post_ln",
        share_weights: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Token Embedding Katmanları
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

        # Pozisyonel Kodlama Katmanları
        self.src_pos = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
        self.tgt_pos = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)

        # Kodlayıcı ve Kod Çözücü Yığınları
        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            norm_type=norm_type,
        )
        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            norm_type=norm_type,
        )

        # Çıkış Jeneratörü (Logit Projeksiyonu)
        self.generator = Generator(d_model=d_model, vocab_size=tgt_vocab_size)

        # Ağırlık Paylaşımı (Weight Tying - Bölüm 3.4)
        if share_weights:
            self.generator.proj.weight = self.tgt_embed.weight
            if src_vocab_size == tgt_vocab_size:
                self.src_embed.weight = self.tgt_embed.weight

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Xavier (Glorot) düzgün dağılımı ile parametre ilklendirme."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Kaynak diziyi kodlar.
        Makale Bölüm 3.4 gereği token embedding tensörü sqrt(d_model) ile çarpılır.
        """
        src_embedded = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.src_pos(src_embedded)
        return self.encoder(x, mask=src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Hedef diziyi ve bellek (memory) tensörünü kullanarak kod çözer.
        """
        tgt_embedded = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.tgt_pos(tgt_embedded)
        return self.decoder(x, memory=memory, src_mask=src_mask, tgt_mask=tgt_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        pad_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Uçtan uca ileri geçiş. Maskeler verilmemişse ve pad_idx tanımlıysa otomatik üretilir.

        Args:
            src (torch.Tensor): Kaynak indeksleri [Batch, Seq_Len_Src]
            tgt (torch.Tensor): Hedef indeksleri [Batch, Seq_Len_Tgt]
            src_mask (Optional[torch.Tensor]): Kaynak maskesi
            tgt_mask (Optional[torch.Tensor]): Hedef maskesi
            pad_idx (Optional[int]): Maskelerin otomatik üretimi için dolgu token indeksi.

        Returns:
            torch.Tensor: Logitler, [Batch, Seq_Len_Tgt, tgt_vocab_size]
        """
        if src_mask is None and tgt_mask is None and pad_idx is not None:
            src_mask, tgt_mask = create_masks(src, tgt, pad_idx=pad_idx)

        memory = self.encode(src, src_mask=src_mask)
        decoder_output = self.decode(tgt, memory=memory, src_mask=src_mask, tgt_mask=tgt_mask)
        logits = self.generator(decoder_output)
        return logits
