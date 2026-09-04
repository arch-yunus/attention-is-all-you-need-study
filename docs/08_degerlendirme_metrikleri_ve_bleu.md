# 08. Değerlendirme Metrikleri ve BLEU Skoru (Evaluation Metrics & BLEU)

> *"In machine translation and sequence modeling, accurate evaluation requires quantifying exact syntactic n-gram overlaps as well as semantic perplexity."*

---

## 1. BLEU Skoru (Bilingual Evaluation Understudy)

Papineni vd. (2002) tarafından önerilen **BLEU**, makine çevirisi çıktısı ile referans çeviri arasındaki n-gram örtüşmesini ölçen standart metriktir.

### 1.1 Matematiksel Formül

$$\text{BLEU} = \text{BP} \cdot \exp\left( \sum_{n=1}^N w_n \ln p_n \right)$$

Burada:
* $p_n$: Modifiye edilmiş $n$-gram hassasiyeti (precision).
* $w_n = 1/N$: Genellikle $N=4$ dereceye kadar eşit ağırlıklar ($w_n = 0.25$).
* $\text{BP}$: Kısalık Cezası (Brevity Penalty).

### 1.2 Kısalık Cezası (Brevity Penalty)

Modelin sadece birkaç yüksek kesinlikli kelime üreterek yüksek hassasiyet elde etmesini engeller:

$$\text{BP} = \begin{cases} 1 & \text{eğer } c > r \\ \exp\left(1 - \frac{r}{c}\right) & \text{eğer } c \le r \end{cases}$$

($c$: Üretilen aday dizinin uzunluğu, $r$: Referans hedef dizinin uzunluğu).

---

## 2. Karmaşıklık (Perplexity - PPL)

Dil modelinin test verisini ne kadar "beklenmedik" veya şaşırtıcı bulduğunu ölçer:

$$\text{PPL} = \exp\left( \mathcal{L}_{\text{Cross-Entropy}} \right) = \exp\left( -\frac{1}{T}\sum_{t=1}^T \log P(x_t \mid x_{<t}) \right)$$

* **Yorum:** $PPL = K$ değeri, modelin her adımda $K$ eşit olasılıklı kelime arasından seçim yaparcasına belirsizlik yaşadığını gösterir. Düşük PPL daima daha iyidir.

---

## 3. Depo İçerisindeki Uygulama

[`src/metrics.py`](file:///g:/Di%C4%9Fer%20bilgisayarlar/Diz%C3%BCst%C3%BC%20Bilgisayar%C4%B1m/github%20repolar%C4%B1m/attention-is-all-you-need-study/src/metrics.py) modülü:

```python
from src import compute_bleu, corpus_bleu, calculate_perplexity, exact_match_accuracy

# Tekil Cümle BLEU
bleu = compute_bleu(reference=[1, 4, 8, 9, 2], candidate=[1, 4, 8, 9, 2])
print(f"BLEU Skoru: {bleu}") # 100.0

# Kayıptan Perplexity
loss = 1.609
ppl = calculate_perplexity(loss)
print(f"Perplexity: {ppl}") # 5.0
```
