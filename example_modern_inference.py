"""
Modern Çıkarım (Inference), KV-Cache ve Dikkat Mimarileri Kıyaslama Demosu.

Bu betik:
1. Greedy vs Beam Search vs Temperature/Top-p Örnekleme yöntemlerini karşılaştırır.
2. KV-Cache hızlandırmasını ölçer ve doğrular.
3. MHA vs GQA vs MQA bellek tasarruflarını hesaplar.
4. Sentetik çeviri üzerinde BLEU skoru ve Perplexity ölçer.
"""

import time
import torch
from src import (
    Transformer,
    greedy_decode,
    beam_search_decode,
    sample_decode,
    compare_attention_memory,
    compute_bleu,
    calculate_perplexity,
)


def main():
    print("=" * 75)
    print("  ATTENTION IS ALL YOU NEED - MODERN ÇIKARIM VE MİMARİ VARYANTLARI")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1] Çalışma Ortamı: {device}")

    # 1. KV-Cache ve Bellek Tasarrufu Analizi (LLaMA 70B Boyutlarında)
    print("\n" + "-" * 75)
    print("[2] Bellek & Bant Genişliği Analizi (32 Katman, 4096 Boyut, 2048 Dizi):")
    print("-" * 75)
    mem_stats = compare_attention_memory(
        batch_size=1,
        seq_len=2048,
        num_heads=32,
        d_model=4096,
        num_layers=32,
    )
    print(f"  • Standart Multi-Head Attention (MHA) KV-Önbellek : {mem_stats['MHA_MB']:>8.2f} MB")
    print(f"  • Grouped-Query Attention (GQA - 8 Grup)         : {mem_stats['GQA_MB']:>8.2f} MB  (Tasarruf: %{mem_stats['GQA_Savings_Pct']})")
    print(f"  • Multi-Query Attention (MQA - 1 Baş)           : {mem_stats['MQA_MB']:>8.2f} MB  (Tasarruf: %{mem_stats['MQA_Savings_Pct']})")

    # 2. Sentetik Model Kurulumu
    vocab_size = 30
    d_model = 128
    num_heads = 4
    num_layers = 3

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=256,
        dropout=0.0,
    ).to(device)
    model.eval()

    # 3. KV-Cache ile Çıkarım Hız Karşılaştırması
    print("\n" + "-" * 75)
    print("[3] KV-Cache vs Standart Çıkarım Hız Kıyaslaması (50 Adım):")
    print("-" * 75)

    src = torch.randint(3, vocab_size, (1, 16), device=device)
    max_steps = 40

    # Standart (KV-Cache olmadan)
    start_t = time.perf_counter()
    res_uncached = greedy_decode(model, src, max_len=max_steps, use_cache=False)
    uncached_time = (time.perf_counter() - start_t) * 1000

    # KV-Cache ile
    start_t = time.perf_counter()
    res_cached = greedy_decode(model, src, max_len=max_steps, use_cache=True)
    cached_time = (time.perf_counter() - start_t) * 1000

    print(f"  • Standart (O(N^2) Yeniden Hesaplama) : {uncached_time:.2f} ms")
    print(f"  • KV-Cache ile (O(N) Adım Başı)       : {cached_time:.2f} ms")
    speedup = uncached_time / max(cached_time, 1e-5)
    print(f"  • Hızlanma Katsayısı                  : {speedup:.2f}x")
    print(f"  • Çıktı Bütünlüğü Doğrulaması        : {'BAŞARILI (Birebir Aynı Tensör)' if torch.equal(res_cached, res_uncached) else 'HATA'}")

    # 4. Kod Çözme Stratejileri Karşılaştırması
    print("\n" + "-" * 75)
    print("[4] Kod Çözme Stratejileri (Greedy vs Beam Search vs Sampling):")
    print("-" * 75)

    res_greedy = greedy_decode(model, src, max_len=12, use_cache=True)
    res_beam = beam_search_decode(model, src, beam_size=4, max_len=12)
    res_sample = sample_decode(model, src, max_len=12, temperature=0.7, top_k=10, top_p=0.9)

    print(f"  Kaynak Dizi   : {src[0].tolist()}")
    print(f"  Greedy Çıktı  : {res_greedy[0].tolist()}")
    print(f"  Beam Search   : {res_beam[0].tolist()} (Işın Boyutu = 4)")
    print(f"  Top-p Sample  : {res_sample[0].tolist()} (T=0.7, p=0.9)")

    # 5. Metrik Hesaplama (BLEU & Perplexity)
    ref = src[0].tolist()
    bleu_greedy = compute_bleu(ref, res_greedy[0].tolist())
    bleu_beam = compute_bleu(ref, res_beam[0].tolist())
    sample_loss = 1.45
    ppl = calculate_perplexity(sample_loss)

    print("\n" + "-" * 75)
    print("[5] Değerlendirme Metrikleri:")
    print("-" * 75)
    print(f"  • Greedy BLEU Skoru      : {bleu_greedy:.2f}")
    print(f"  • Beam Search BLEU Skoru : {bleu_beam:.2f}")
    print(f"  • Örnek Kayıp (Loss 1.45): Perplexity (PPL) = {ppl:.3f}")
    print("\n" + "=" * 75)


if __name__ == "__main__":
    main()
