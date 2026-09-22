"""
Kapsamlı Dikkat ve Çıkarım Benchmark Paketi (Comprehensive Benchmark Suite).

Bu script, aşağıdaki mimari varyantlarının performansını karşılaştırır:
1. Standart Çok Başlı Dikkat (MHA - Multi-Head Attention)
2. Gruplanmış Sorgu Dikkati (GQA - Grouped-Query Attention)
3. Çoklu Sorgu Dikkati (MQA - Multi-Query Attention)
4. ALiBi Dikkati (Attention with Linear Biases)
5. Kayan Pencere Dikkati (Sliding Window Attention)
6. PyTorch Flash / SDPA Dikkati
7. KV-Cache vs No-Cache Çıkarım Gecikmesi (Latency / Tokens-per-second)
"""

import sys
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
from src.multi_head_attention import MultiHeadAttention
from src.attention_variants import (
    MultiQueryAttention,
    GroupedQueryAttention,
    ALiBiAttention,
    SlidingWindowAttention,
    FlashAttentionSDPA,
    compare_attention_memory,
)
from src.decoder_only import DecoderOnlyTransformer


def benchmark_forward(layer: nn.Module, x: torch.Tensor, warmup: int = 5, reps: int = 20) -> float:
    """Bir katmanın ileri geçiş süresini (milisaniye cinsinden) ölçer."""
    with torch.no_grad():
        for _ in range(warmup):
            if isinstance(layer, SlidingWindowAttention):
                _ = layer(x)
            elif isinstance(layer, FlashAttentionSDPA):
                _ = layer(x, x, x)
            else:
                _ = layer(x, x, x)

        start = time.perf_counter()
        for _ in range(reps):
            if isinstance(layer, SlidingWindowAttention):
                _ = layer(x)
            elif isinstance(layer, FlashAttentionSDPA):
                _ = layer(x, x, x)
            else:
                _ = layer(x, x, x)
        elapsed = (time.perf_counter() - start) / reps * 1000.0
    return elapsed


def main():
    print("=" * 80)
    print("[*] Transformer & Dikkat Varyantlari Kapsamli Benchmark Paketi")
    print("=" * 80)

    d_model = 512
    num_heads = 8
    seq_len = 512
    batch_size = 4
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\n[Test Konfigurasyonu]: Batch: {batch_size}, Seq_Len: {seq_len}, d_model: {d_model}, Heads: {num_heads}\n")

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    gqa = GroupedQueryAttention(d_model=d_model, num_heads=num_heads, num_kv_heads=2, dropout=0.0)
    mqa = MultiQueryAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    alibi = ALiBiAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    swa = SlidingWindowAttention(d_model=d_model, num_heads=num_heads, window_size=128, dropout=0.0)
    sdpa = FlashAttentionSDPA(d_model=d_model, num_heads=num_heads, dropout=0.0)

    modules = [
        ("MHA (Vaswani 2017)", mha),
        ("GQA (LLaMA-3 / 2 KV Heads)", gqa),
        ("MQA (PaLM / Falcon / 1 KV Head)", mqa),
        ("ALiBi (BLOOM / No Pos Emb)", alibi),
        ("SWA (Mistral 7B / Window 128)", swa),
        ("PyTorch SDPA (FlashAttention)", sdpa),
    ]

    print(f"{'Varyant Adi':<35} | {'Gecikme (ms)':<15} | {'Throughput (Tokens/s)':<22}")
    print("-" * 80)

    for name, mod in modules:
        mod.eval()
        ms = benchmark_forward(mod, x)
        tokens_per_sec = (batch_size * seq_len) / (ms / 1000.0)
        print(f"{name:<35} | {ms:>10.3f} ms | {tokens_per_sec:>18,.0f} tok/s")

    # KV-Cache Bellek Analizi
    print("\n" + "=" * 80)
    print("[*] 70B Olcekli Modelde KV-Cache Bellek Tuketimi (32 Katman, FP16)")
    print("=" * 80)

    mem_stats = compare_attention_memory(
        batch_size=1,
        seq_len=4096,
        num_heads=32,
        d_model=4096,
        num_layers=32,
    )
    print(f" - MHA Bellek (32 Heads) : {mem_stats['MHA_MB']:>8.1f} MB (Referans)")
    print(f" - GQA Bellek (8 Groups)  : {mem_stats['GQA_MB']:>8.1f} MB ({mem_stats['GQA_Savings_Pct']}% Tasarruf)")
    print(f" - MQA Bellek (1 Head)    : {mem_stats['MQA_MB']:>8.1f} MB ({mem_stats['MQA_Savings_Pct']}% Tasarruf)")

    # Decoder-Only KV-Cache Çıkarım Hızı Kıyaslaması
    print("\n" + "=" * 80)
    print("[*] Decoder-Only LLM Cikarim Hizi: KV-Cache vs No-Cache")
    print("=" * 80)

    mini_llm = DecoderOnlyTransformer(
        vocab_size=256,
        d_model=128,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=512,
    )
    mini_llm.eval()
    prompt = torch.randint(0, 256, (1, 32), dtype=torch.long)

    # Cache ile
    start = time.perf_counter()
    _ = mini_llm.generate(prompt, max_new_tokens=25, use_cache=True)
    cache_time = time.perf_counter() - start

    # Cache olmadan
    start = time.perf_counter()
    _ = mini_llm.generate(prompt, max_new_tokens=25, use_cache=False)
    no_cache_time = time.perf_counter() - start

    speedup = no_cache_time / max(cache_time, 1e-6)
    print(f" - KV-Cache Ile Uretim Suresi   : {cache_time * 1000.0:.2f} ms ({25 / cache_time:.1f} tok/s)")
    print(f" - KV-Cache Olmadan Sure        : {no_cache_time * 1000.0:.2f} ms ({25 / no_cache_time:.1f} tok/s)")
    print(f" - Hizlanma Orani (Speedup)      : {speedup:.2f}x Daha Hizli!")

    print("\n" + "=" * 80)
    print("[SUCCESS] Benchmark Basariyla Tamamlandi!")
    print("=" * 80)


if __name__ == "__main__":
    main()
