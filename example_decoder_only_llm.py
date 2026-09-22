"""
Modern Salt-Dekoder (Decoder-Only) LLM Eğitimi ve Çıkarım Örneği.

Bu script, modern LLM mimarilerini (LLaMA-3, Mistral) simüle eder:
1. Metin külliyatı üzerinden saf Python BPE Tokenizer eğitilir.
2. RoPE, RMSNorm, SwiGLU ve GQA içeren modern Decoder-Only Transformer inşa edilir.
3. Model mini veri kümesi üzerinde eğitilir (CrossEntropyLoss).
4. KV-Cache ve Min-P / Top-P / Repetition Penalty ile hızlı çıkarım gerçekleştirilir.
"""

import sys
import os

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.optim as optim
from src.tokenizer import BPETokenizer
from src.decoder_only import DecoderOnlyTransformer


def main():
    print("=" * 75)
    print("[*] Modern Decoder-Only LLM (RoPE + RMSNorm + SwiGLU + GQA) Egitimi & Cikarimi")
    print("=" * 75)

    # 1. Eğitim Metin Külliyatı (Corpus)
    training_corpus = [
        "Dikkat mekanizmasi derin ogrenmede devrim yaratti.",
        "Transformer mimarisi dogal dil islemenin temel tasidir.",
        "GQA ve RoPE modern buyuk dil modellerinde standarttir.",
        "KV-Cache cikarim sirasinda bellek ve islem tasarrufu saglar.",
        "RMSNorm ve SwiGLU gradyan kararliligini ve model kalitesini artirir.",
        "Min-P ornekleme yontemi halusinasyonu azaltip dogal metin uretir.",
        "Buyuk dil modelleri milyarlarca parametre ile dunyayi anlar.",
        "Yapay zeka gelecegin teknolojilerini sekillendiriyor.",
    ]

    print(f"\n[1/4] BPE Tokenizer Egitiliyor ({len(training_corpus)} cumle)...")
    tokenizer = BPETokenizer()
    tokenizer.train(training_corpus, target_vocab_size=150, min_frequency=1)
    print(f"[+] Sozluk boyutu: {tokenizer.vocab_size} token")

    # 2. Decoder-Only Model Yapılandırması
    d_model = 128
    num_heads = 4
    num_kv_heads = 2  # GQA: 4 sorgu başı, 2 KV başı (2x tasarruf)
    num_layers = 3
    max_seq_len = 64

    print(f"\n[2/4] Decoder-Only LLM Modeli Insa Ediliyor...")
    print(f" - Gizli Boyut (d_model): {d_model}")
    print(f" - Dikkat Baslari: {num_heads} Query Heads, {num_kv_heads} KV Heads (GQA)")
    print(f" - Katman Sayisi: {num_layers}")
    print(f" - Aktivasyon: SwiGLU | Normalizasyon: RMSNorm | Pozisyon: RoPE")

    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_seq_len=max_seq_len,
        norm_type="rmsnorm",
        ffn_type="swiglu",
        dropout=0.0,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Toplam Egitilebilir Parametre Sayisi: {total_params:,}")

    # 3. Veri Hazırlama ve Model Eğitimi
    input_ids, _ = tokenizer.encode_batch(training_corpus, max_len=20, add_special_tokens=True)
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    print(f"\n[3/4] Model Egitiliyor (Epochs: 60)...")
    model.train()

    for epoch in range(1, 61):
        optimizer.zero_grad()
        logits, loss = model(inputs, targets=targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 15 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/60 | Loss: {loss.item():.4f}")

    # 4. Hızlı Çıkarım ve Üretim Testi (KV-Cache + Min-P Sampling)
    print("\n[4/4] KV-Cache ve Min-P Sampling ile Metin Uretimi Test Ediliyor...")
    model.eval()

    test_prompts = [
        "Transformer mimarisi",
        "KV-Cache cikarim",
        "Yapay zeka",
    ]

    for prompt in test_prompts:
        prompt_tokens = torch.tensor([tokenizer.encode(prompt, add_special_tokens=True)], dtype=torch.long)
        
        generated_tokens = model.generate(
            prompt_tokens=prompt_tokens,
            max_new_tokens=15,
            temperature=0.8,
            top_k=20,
            top_p=0.9,
            min_p=0.05,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        decoded_text = tokenizer.decode(generated_tokens[0].tolist(), skip_special_tokens=True)
        print(f"\n[>] Istem   : \"{prompt}\"")
        print(f"[+] Uretilen: \"{decoded_text}\"")

    print("\n" + "=" * 75)
    print("[SUCCESS] Modern Decoder-Only LLM Pipeline Basariyla Tamamlandi!")
    print("=" * 75)


if __name__ == "__main__":
    main()
