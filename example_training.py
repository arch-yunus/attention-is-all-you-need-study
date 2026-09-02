"""
Uçtan Uca Transformer Eğitim ve Çıkarım Örneği (End-to-End Training & Inference Demo).

Bu betik, referans Transformer mimarisini sentetik bir dizi kopyalama/öğrenme
görevi (Copy Task) üzerinde eğitir, Noam zamanlayıcısını ve Label Smoothing kaybını kullanır,
ardından eğitilen modelle otoregresif açgözlü çıkarım (greedy decoding) yapar.
"""

import time
import torch
import torch.nn as nn
from src import (
    Transformer,
    create_masks,
    generate_square_subsequent_mask,
    NoamLR,
    LabelSmoothingLoss,
)

# Özel Token Tanımları
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
VOCAB_SIZE = 25  # 0: PAD, 1: BOS, 2: EOS, 3-24: Kelimeler


def generate_batch(batch_size: int, seq_len: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sentetik kopyalama görevi verisi üretir:
    Kaynak: [BOS, w_1, w_2, ..., w_k, EOS]
    Hedef Girdi (Decoder In): [BOS, w_1, w_2, ..., w_k]
    Hedef Çıktı (Expected Out): [w_1, w_2, ..., w_k, EOS]
    """
    # 3 ile VOCAB_SIZE-1 arasında rastgele tokenlar
    data = torch.randint(3, VOCAB_SIZE, (batch_size, seq_len))
    bos = torch.full((batch_size, 1), BOS_IDX, dtype=torch.long)
    eos = torch.full((batch_size, 1), EOS_IDX, dtype=torch.long)

    src = torch.cat([bos, data, eos], dim=1)
    tgt_in = torch.cat([bos, data], dim=1)
    tgt_expected = torch.cat([data, eos], dim=1)

    return src, tgt_in, tgt_expected


def greedy_decode(model: Transformer, src: torch.Tensor, max_len: int = 12) -> torch.Tensor:
    """
    Eğitilmiş model ile otoregresif adım adım greedy çıkarım yapar.
    """
    model.eval()
    batch_size = src.size(0)
    device = src.device

    src_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)
    memory = model.encode(src, src_mask=src_mask)

    # Başlangıç tokenı <BOS>
    ys = torch.full((batch_size, 1), BOS_IDX, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(ys.size(1), device=device).unsqueeze(1)
        out = model.decode(ys, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        prob = model.generator(out[:, -1])
        next_word = torch.argmax(prob, dim=-1, keepdim=True)
        ys = torch.cat([ys, next_word], dim=1)

        # Tüm batch'teki örnekler EOS ürettiyse erken dur
        if (ys == EOS_IDX).any(dim=1).all():
            break

    return ys


def main():
    print("=" * 65)
    print("  Transformer Referans Mimarisi: Sentetik Eğitim ve Çıkarım")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Çalışma Aygıtı: {device}\n")

    # Hafif eğitim için kompakt model konfigürasyonu
    d_model = 64
    num_heads = 4
    num_layers = 2
    d_ff = 128
    epochs = 15
    batch_size = 32
    steps_per_epoch = 20

    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
        dropout=0.1,
        share_weights=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Eğitilebilir Parametre Sayısı: {total_params:,}")

    # Optimizer, Noam Scheduler ve Label Smoothing
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamLR(optimizer, d_model=d_model, warmup_steps=100, factor=1.5)
    criterion = LabelSmoothingLoss(vocab_size=VOCAB_SIZE, padding_idx=PAD_IDX, smoothing=0.1)

    print("\nEğitim Başlatılıyor...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for _ in range(steps_per_epoch):
            src, tgt_in, tgt_expected = generate_batch(batch_size, seq_len=6)
            src, tgt_in, tgt_expected = src.to(device), tgt_in.to(device), tgt_expected.to(device)

            src_mask, tgt_mask = create_masks(src, tgt_in, pad_idx=PAD_IDX)

            optimizer.zero_grad()
            logits = model(src, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask)
            loss = criterion(logits, tgt_expected)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / steps_per_epoch
        lr = scheduler.get_last_lr()[0]
        if epoch % 3 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:2d}/{epochs:2d}] | Ortalama Kayıp: {avg_loss:.4f} | LR: {lr:.6f}")

    elapsed = time.time() - start_time
    print(f"\nEğitim Tamamlandı! ({elapsed:.2f} saniye)\n")

    # Çıkarım (Inference) Testi
    print("-" * 65)
    print("Test Çıkarımı (Otoregresif Greedy Decoding):")
    print("-" * 65)

    test_src, _, test_expected = generate_batch(batch_size=3, seq_len=6)
    test_src = test_src.to(device)
    decoded_preds = greedy_decode(model, test_src, max_len=10)

    for i in range(3):
        src_tokens = [int(t) for t in test_src[i].cpu() if t not in (PAD_IDX, BOS_IDX, EOS_IDX)]
        pred_tokens = [int(t) for t in decoded_preds[i].cpu() if t not in (PAD_IDX, BOS_IDX, EOS_IDX)]
        print(f"Örnek {i+1}:")
        print(f"  Kaynak Girdi     : {src_tokens}")
        print(f"  Modelin Çıktısı  : {pred_tokens}")
        is_success = "TAM EŞLEŞME" if src_tokens == pred_tokens else "ÖĞRENİYOR"
        print(f"  Durum            : {is_success}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
