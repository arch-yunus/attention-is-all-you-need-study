# 07. Modern Dikkat Mimarileri: MQA, GQA ve RoPE

> *"The evolution of Attention from 2017 to modern LLMs is the story of conquering memory bandwidth bottlenecks and achieving superior positional extrapolation."*

---

## 1. Dikkat Mekanizmalarının Karşılaştırması

2017 orijinal makalesindeki **Multi-Head Attention (MHA)**, çıkarım (inference) anında her katmanda $H$ adet Key ve Value tensörünü saklamayı gerektirir. Uzun bağlam pencerelerinde (8k, 32k, 128k token) bu durum GPU VRAM darboğazına yol açar.

```text
1. Multi-Head Attention (MHA - Vaswani vd., 2017):
   Q Başları: [Q1] [Q2] [Q3] [Q4] [Q5] [Q6] [Q7] [Q8]
   K Başları: [K1] [K2] [K3] [K4] [K5] [K6] [K7] [K8]
   V Başları: [V1] [V2] [V3] [V4] [V5] [V6] [V7] [V8]

2. Multi-Query Attention (MQA - Shazeer, 2019):
   Q Başları: [Q1] [Q2] [Q3] [Q4] [Q5] [Q6] [Q7] [Q8]
   K Başları: [------------- Tek Bir K Başı -------------]
   V Başları: [------------- Tek Bir V Başı -------------]

3. Grouped-Query Attention (GQA - Ainslie vd., 2023 / LLaMA-2/3):
   Q Başları: [Q1, Q2] [Q3, Q4] [Q5, Q6] [Q7, Q8]
   K Başları:   [K_G1]   [K_G2]   [K_G3]   [K_G4]
   V Başları:   [V_G1]   [V_G2]   [V_G3]   [V_G4]
```

### 1.1 Bellek ve Hesaplama Kıyaslama Tablosu

| Mimari | Sorgu (Q) Baş Sayısı | Anahtar/Değer (KV) Baş Sayısı | KV-Cache Bellek Maliyeti | Kalite / Model Doğruluğu | Kullanıldığı Modeller |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **MHA** | $H$ | $H$ | $1.0\times$ (Referans) | En Yüksek | Vaswani 2017, GPT-3, BERT, T5 |
| **GQA** | $H$ | $G$ ($1 < G < H$) | $\frac{G}{H}\times$ (%75 - %87.5 Tasarruf) | MHA ile Neredeyse Eşit | LLaMA-2/3 (70B), Mistral, Gemma 2 |
| **MQA** | $H$ | $1$ | $\frac{1}{H}\times$ (%96.9 Tasarruf) | Hafif Düşüş | PaLM, StarCoder, Falcon |

---

## 2. Döner Pozisyonel Gömme (Rotary Position Embedding - RoPE)

Su vd. (2021) tarafından önerilen **RoPE**, pozisyonel bilgiyi mutlak toplam vektörü olarak eklemek yerine, sorgu ve anahtar vektörlerini 2 boyutlu alt uzaylarda açısıyla döndürür:

$$\mathbf{R}_{\Theta, m}^d = \text{diag}\left( \mathbf{R}_{\theta_1, m}, \mathbf{R}_{\theta_2, m}, \dots, \mathbf{R}_{\theta_{d/2}, m} \right)$$

$$\mathbf{R}_{\theta_i, m} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$$

### RoPE'un Temel Üstünlükleri:
1. **Göreceli Pozisyon Özelliği:** İki token arasındaki iç çarpım, doğrudan $m - n$ bağıl mesafesine bağlı hale gelir:
   $$\langle \mathbf{R}_m \mathbf{q}, \mathbf{R}_n \mathbf{k} \rangle = g(\mathbf{q}, \mathbf{k}, m-n)$$
2. **Uzun Bağlam Ekstrapolasyonu:** Model eğitim uzunluğunun ötesindeki dizilere sıfır hata ile genelleme yapabilir (RoPE Scaling / YaRN).
3. **Üniter Matris Özelliği:** Vektör normunu korur, gradyan patlamasını engeller.

---

## 3. Depo Kod Kullanımı

[`src/attention_variants.py`](file:///g:/Di%C4%9Fer%20bilgisayarlar/Diz%C3%BCst%C3%BC%20Bilgisayar%C4%B1m/github%20repolar%C4%B1m/attention-is-all-you-need-study/src/attention_variants.py) modülü üzerinden tüm modern varyantları doğrudan kullanabilirsiniz:

```python
from src import MultiQueryAttention, GroupedQueryAttention, RotaryPositionalEmbedding

# 8 Sorgu Başı, 2 KV Başı içeren Grouped-Query Attention
gqa = GroupedQueryAttention(d_model=512, num_heads=8, num_kv_heads=2)
out, attn = gqa(q, k, v)

# RoPE Açılı Dönüşümü
rope = RotaryPositionalEmbedding(dim=64, max_seq_len=4096)
q_rot, k_rot = rope(q, k)
```
