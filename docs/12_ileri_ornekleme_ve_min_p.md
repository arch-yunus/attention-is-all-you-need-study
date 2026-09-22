# Bölüm 12: İleri Seviye Çıkarım, Örnekleme ve Min-P Algoritması

> *"Min-P sampling establishes a dynamic cutoff proportional to the probability of the most likely token, eliminating the low-probability junk without cutting off valid alternatives in ambiguous contexts."*  
> — **Grimminger et al.**, *"Min-P Sampling: A Better Alternative to Top-P for LLM Inference" (2024)*

---

## 1. Kod Çözme ve Örnekleme Problemi

Eğitilmiş bir dil modeli her adımda kelime haznesindeki tüm tokenlar için bir logit dağılımı üretir:

$$P(y_t \mid y_{<t}, x) = \text{softmax}(z_t)$$

- **Greedy Decoding**: Her zaman $\arg\max$ seçer. Sonuç deterministiktir ancak tekrara (repetition loop) ve kuru bir üsluba yol açar.
- **Top-K**: En yüksek $K$ tokenı korur. Ancak düz dağılımlarda çok dar, sivri dağılımlarda ise çok geniş kalabilir.
- **Top-P (Nucleus)**: Kümülatif toplamı $P$ olan çekirdek kümeyi seçer. Sivri dağılımlarda doğru çalışırken, belirsiz durumlarda halüsinasyona açık düşük kaliteli tokenları da dahil edebilir.

---

## 2. Min-P Örnekleme Mekanizması

Min-P filtresi, en olası tokenın olasılığına ($p_{\max}$) bağlı olarak dinamik bir alt sınır belirler:

$$\text{Eşik}_{\text{min\_p}} = p_{\max} \cdot \text{min\_p}$$

Sadece $P(y_i) \ge \text{Eşik}_{\text{min\_p}}$ şartını sağlayan tokenlar örnekleme havuzunda tutulur, diğerlerinin logitleri $-\infty$ yapılır.

### Örnek Senaryo

1. **Yüksek Güven Durumu** ($p_{\max} = 0.90$, $\text{min\_p} = 0.05$):
   - Eşik = $0.90 \times 0.05 = 0.045$.
   - Olasılığı %4.5 altındaki tüm gürültülü tokenlar anında elenir.
2. **Düşük Güven / Yaratıcı Durum** ($p_{\max} = 0.20$, $\text{min\_p} = 0.05$):
   - Eşik = $0.20 \times 0.05 = 0.010$.
   - Model %1'in üzerindeki tüm alternatif fikirleri havuzda tutar.

---

## 3. Tekrar ve Frekans Cezaları

- **Repetition Penalty (Keskar et al., 2019)**: Önceki bağlamda geçen tokenların logitlerini $z_i \to z_i / \theta$ veya $z_i \cdot \theta$ ($z_i < 0$ ise) yaparak cezalandırır.
- **Frequency Penalty**: Bir token bağlamda kaç kez geçiyorsa o kadar orantılı logit düşüşü sağlar: $z_i \to z_i - \alpha \cdot \text{count}(i)$.
