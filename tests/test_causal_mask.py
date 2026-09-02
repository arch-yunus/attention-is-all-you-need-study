"""
Gelecek Sızıntısı (Causal Leakage) Doğrulama Testi.

Causal/Look-ahead maskelemenin temel görevi:
t anındaki tahminin t+1, t+2, ... gibi gelecekteki token'ların değerinden
asla ve kesinlikle etkilenmemesini (bilgi sızmamasını) garanti etmektir.
"""

import pytest
import torch
from src import Transformer, create_masks, generate_square_subsequent_mask


def test_causal_mask_structure():
    sz = 5
    mask = generate_square_subsequent_mask(sz).squeeze(0)
    # Üst üçgen False (0), alt üçgen True (1) olmalıdır
    for i in range(sz):
        for j in range(sz):
            if j <= i:
                assert mask[i, j].item() is True, f"({i}, {j}) konumu True olmalıydı"
            else:
                assert mask[i, j].item() is False, f"({i}, {j}) konumu False (maskeli) olmalıydı"


def test_no_future_leakage_in_decoder():
    """
    Kritik Test: Gelecekteki token'lar değiştirildiğinde,
    önceki adımların ürettiği çıktılar %100 AYNI kalmalıdır.
    """
    torch.manual_seed(42)
    vocab_size = 50
    d_model = 64
    seq_len = 6

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        dropout=0.0,  # Deterministik sonuç için dropout 0
    )
    model.eval()

    src = torch.randint(1, vocab_size, (1, 5))

    # İki farklı hedef dizi: İlk 3 token birebir aynı, son 3 token tamamen farklı
    tgt_prefix = torch.tensor([[10, 20, 30]])
    tgt_suffix_A = torch.tensor([[40, 41, 42]])
    tgt_suffix_B = torch.tensor([[5, 8, 9]])

    tgt_A = torch.cat([tgt_prefix, tgt_suffix_A], dim=1)  # [10, 20, 30, 40, 41, 42]
    tgt_B = torch.cat([tgt_prefix, tgt_suffix_B], dim=1)  # [10, 20, 30,  5,  8,  9]

    with torch.no_grad():
        src_mask_A, tgt_mask_A = create_masks(src, tgt_A)
        src_mask_B, tgt_mask_B = create_masks(src, tgt_B)

        out_A = model(src, tgt_A, src_mask=src_mask_A, tgt_mask=tgt_mask_A)
        out_B = model(src, tgt_B, src_mask=src_mask_B, tgt_mask=tgt_mask_B)

    # 0., 1. ve 2. pozisyonlardaki logitler birebir eşit olmalıdır!
    # Gelecekteki tokenların değişmesi geçmişteki tahminleri etkileyemez!
    torch.testing.assert_close(
        out_A[:, :3, :],
        out_B[:, :3, :],
        rtol=1e-5,
        atol=1e-5,
        msg="Causal maskeleme hatası: Gelecekteki tokenlar geçmiş adımlara sızdı!",
    )

    # Sonraki pozisyonlar (3, 4, 5) ise farklı olmalıdır
    diff = (out_A[:, 3:, :] - out_B[:, 3:, :]).abs().sum()
    assert diff > 1e-3, "Farklı girdilere rağmen sonraki pozisyonlar değişmedi!"
