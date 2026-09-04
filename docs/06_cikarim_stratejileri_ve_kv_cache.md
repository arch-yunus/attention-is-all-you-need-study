# 06. Çıkarım Stratejileri ve KV-Cache (Inference & KV-Cache)

> *"Inference in autoregressive transformers is memory-bandwidth bound, not compute bound. KV-Caching turns an $O(N^2)$ repetitive projection bottleneck into an $O(N)$ append operation, unlocking real-time generation."*

---

## 1. Otoregresif Üretim Darboğazı ve KV-Cache

Transformer Kod Çözücüsü (Decoder), sonraki kelimeyi tahmin etmek için önceki tüm adımların çıktılarını girdi olarak alır. 

### 1.1 Klasik (Naive) Çıkarım vs KV-Cache

* **Klasik Yaklaşım ($O(N^2)$ İşlem):**  
  Her yeni $t$ zaman adımında, $1 \dots t$ arasındaki tüm tokenlar embedding ve doğrusal projeksiyonlardan ($W^Q, W^K, W^V$) tekrar geçirilir. Bu durum adım sayısı arttıkça karesel hesaplama yükü oluşturur.
* **KV-Cache ($O(N)$ İşlem):**  
  Önceki adımlarda hesaplanan $K$ ve $V$ tensörleri GPU belleğinde (VRAM) saklanır. $t$ anında yalnızca yeni gelen $x_t$ token'ı için $Q_t, K_t, V_t$ hesaplanır ve $K_t, V_t$ önbelleğe eklenir (concatenate).

```text
KLASİK ÇIKARIM (Her adımda baştan hesaplama):
Adım 1: Q1 x [K1]^T
Adım 2: Q2 x [K1, K2]^T           <-- K1 tekrar hesaplandı
Adım 3: Q3 x [K1, K2, K3]^T       <-- K1, K2 tekrar hesaplandı

KV-CACHE İLE ÇIKARIM (Önbellekten birleştirme):
Adım 1: Q1 x [K1]^T               --> [K1, V1] önbelleğe yaz
Adım 2: Q2 x [K1, K2]^T           --> Yalnızca K2 hesaplandı, önbelleğe eklendi
Adım 3: Q3 x [K1, K2, K3]^T       --> Yalnızca K3 hesaplandı, önbelleğe eklendi
```

---

## 2. Kod Çözme Stratejileri (Decoding Strategies)

Modelin ürettiği logit dağılımından bir sonraki token'ın nasıl seçileceği metin akıcılığı ve doğruluğu açısından kritiktir.

### 2.1 Açgözlü Çıkarım (Greedy Decoding)
Her adımda en yüksek olasılığa sahip token seçilir:
$$y_t = \arg\max_{w \in \mathcal{V}} P(w \mid y_{<t}, x)$$
* **Avantajı:** Çok hızlı ve deterministiktir.
* **Dezavantajı:** Erken bir yanlış seçim kümülatif hatalara ve tekrarlara (repetition loops) yol açar.

### 2.2 Işın Araması (Beam Search)
Her adımda tek bir en iyi kelime yerine, kümülatif olasılığı en yüksek $B$ (beam size) adet hipotez takip edilir:
$$\text{Score}(Y) = \frac{\sum_{t=1}^{|Y|} \log P(y_t \mid y_{<t}, x)}{LP(|Y|)}$$

Burada $LP(|Y|)$ uzunluk cezası (Length Penalty) olup kısa cümleleri kayırmayı engeller (Wu vd., 2016):
$$LP(|Y|) = \left(\frac{5 + |Y|}{6}\right)^\alpha \quad (\alpha \in [0.6, 0.8])$$

### 2.3 Olasılıksal Örnekleme (Sampling) & Sıcaklık (Temperature)
Deterministik seçim yerine olasılık dağılımından rastgele çekim yapılır:
$$P(y_t = w) = \frac{\exp(z_w / T)}{\sum_{j} \exp(z_j / T)}$$
* $T < 1.0$: Dağılımı sivrileştirir (daha tutarlı ve kesin seçimler).
* $T > 1.0$: Dağılımı düzleştirir (daha yaratıcı ve çeşitli çıktılar).

### 2.4 Top-K ve Top-P (Nucleus) Filtreleme
* **Top-K (Fan vd., 2018):** Sadece en yüksek olasılıklı $K$ token çekim havuzunda tutulur.
* **Top-P / Nucleus (Holtzman vd., 2019):** Kümülatif toplam olasılığı $p$ (örn: 0.90) eşiğini aşan dinamik token alt kümesi seçilir.

---

## 3. Depo İçerisindeki Uygulama

Bu algoritmalar depomuzda [`src/generation.py`](file:///g:/Di%C4%9Fer%20bilgisayarlar/Diz%C3%BCst%C3%BC%20Bilgisayar%C4%B1m/github%20repolar%C4%B1m/attention-is-all-you-need-study/src/generation.py) içerisinde KV-Cache destekli olarak sunulmuştur:

```python
from src import Transformer, greedy_decode, beam_search_decode, sample_decode

# KV-Cache destekli hızlı açgözlü çıkarım
output_greedy = greedy_decode(model, src, max_len=30, use_cache=True)

# Uzunluk cezalı Işın Araması
output_beam = beam_search_decode(model, src, beam_size=4, max_len=30, length_penalty_alpha=0.6)

# Nucleus (Top-p) ve Sıcaklık Örneklemesi
output_sample = sample_decode(model, src, max_len=30, temperature=0.7, top_p=0.9)
```
