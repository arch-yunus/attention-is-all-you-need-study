"""
Maskeleme Yardımcı Fonksiyonları (Mask Utilities).

Transformer modelinde iki temel maske kullanılır:
1. Dolgu Maskesi (Padding Mask): <PAD> token'larının dikkat hesaplamasına katılmasını engeller.
2. Nedensel/Gelecek Maskesi (Causal / Look-Ahead Mask): Decoder'ın autoregressive üretimde
   gelecekteki token'ları görmesini engeller.
"""

from typing import Tuple
import torch


def generate_square_subsequent_mask(sz: int, device: torch.device = None) -> torch.Tensor:
    """
    Kare Causal (Look-Ahead) maskesi üretir.
    Üst üçgen sıfırlanır (veya False yapılır), alt üçgen ve köşegen 1 (veya True) kalır.

    Args:
        sz (int): Dizi uzunluğu (Seq_Len).
        device (torch.device, optional): Tensörün yerleştirileceği aygıt.

    Returns:
        torch.Tensor: [1, sz, sz] şeklinde alt üçgen boolean tensörü.
    """
    # 1: izin verilen (geçmiş ve şimdiki zaman), 0: engellenen (gelecek)
    mask = torch.tril(torch.ones((sz, sz), device=device, dtype=torch.bool)).unsqueeze(0)
    return mask


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Padding token'larını maskeleyen tensör üretir.

    Args:
        seq (torch.Tensor): Token indeksleri tensörü, şekil: [Batch, Seq_Len]
        pad_idx (int): Dolgu token'ının indeks değeri. Varsayılan: 0.

    Returns:
        torch.Tensor: [Batch, 1, 1, Seq_Len] şeklinde boolean maske (True: geçerli token, False: PAD).
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_masks(
    src: torch.Tensor,
    tgt: torch.Tensor,
    pad_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Kaynak ve hedef diziler için tüm maskeleri tek seferde üretir.

    Args:
        src (torch.Tensor): Kaynak token indeksleri, [Batch, Seq_Len_Src]
        tgt (torch.Tensor): Hedef token indeksleri, [Batch, Seq_Len_Tgt]
        pad_idx (int): Dolgu indeksi.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - src_mask: [Batch, 1, 1, Seq_Len_Src]
            - tgt_mask: [Batch, 1, Seq_Len_Tgt, Seq_Len_Tgt] (Causal + Padding bileşimi)
    """
    src_mask = create_padding_mask(src, pad_idx=pad_idx)

    tgt_pad_mask = create_padding_mask(tgt, pad_idx=pad_idx)  # [Batch, 1, 1, Seq_Len_Tgt]
    tgt_seq_len = tgt.size(1)
    causal_mask = generate_square_subsequent_mask(tgt_seq_len, device=tgt.device)  # [1, Seq_Len_Tgt, Seq_Len_Tgt]

    # İki maskenin mantıksal VE (AND) kesişimi
    tgt_mask = tgt_pad_mask & causal_mask.unsqueeze(1)  # [Batch, 1, Seq_Len_Tgt, Seq_Len_Tgt]

    return src_mask, tgt_mask
