# 04. Encoder-Decoder Köprüsü: Çapraz Dikkat (Cross-Attention) ve Otoregresif Çıkarım

> *"Kod çözücüdeki alt katmanlardan birinde, konumların sonraki konumlara dikkat yöneltmesini engelleyecek şekilde düzenleme yaptık... Bu maskeleme, $i$ konumu için yapılan tahminlerin yalnızca $i$'den küçük konumlardaki bilinen çıktılara bağlı olmasını güvenceye alır."*  
> — **Vaswani et al., 2017 (Bölüm 3.1)**

---

## 1. Encoder ve Decoder'ın Rol Dağılımı

Orijinal 2017 Transformer mimarisi bir **Dizi Dönüşüm (Sequence-to-Sequence)** modelidir (örn. İngilizce $\to$ Almanca çevirisi). Bu yapı iki ana organdan oluşur:

1. **Kodlayıcı (Encoder):** Kaynak cümlenin tüm kelimelerini eş zamanlı olarak okur, çift yönlü (bidirectional) bağlamsal ilişkileri kurar ve zengin bir semantik bellek (`memory`) tensörü üretir:
   $$\text{Memory} \in \mathbb{R}^{B \times S_{\text{src}} \times d_{\text{model}}}$$
2. **Kod Çözücü (Decoder):** Kaynak belleği ve şimdiye kadar üretilmiş hedef kelimeleri okuyarak bir sonraki kelimeyi tahmin eder (tek yönlü / autoregressive).

---

## 2. Çapraz Dikkat (Cross-Attention) Mekaniği

Decoder katmanının en kritik unsuru, Encoder ile kurduğu **Cross-Attention** köprüsüdür. Öz-dikkatten (Self-Attention) temel farkı girdi kaynaklarıdır:

| Dikkat Türü | Sorgu ($Q$) | Anahtar ($K$) | Değer ($V$) | Amaç |
| :--- | :--- | :--- | :--- | :--- |
| **Encoder Self-Attention** | Encoder | Encoder | Encoder | Kaynak kelimelerin kendi aralarındaki bağlamı |
| **Decoder Self-Attention** | Decoder | Decoder | Decoder | Hedef kelimelerin geçmiş bağlamı (Causal Maskeli) |
| **Cross-Attention** | **Decoder** | **Encoder** | **Encoder** | "Hedef kelime üretilirken kaynak cümlenin neresine bakılmalı?" |

```
Decoder Temsili (x) ──────────────> Projeksiyon (W_Q) ──────────> Sorgu (Q)  [B, H, S_tgt, d_k]
                                                                        │
                                                                   [ Matris Çarpımı ] ──> Benzerlik Matrisi
                                                                        ▲                  [B, H, S_tgt, S_src]
Encoder Çıktısı (Memory) ────┬────> Projeksiyon (W_K) ──────────> Anahtar (K)│ [B, H, S_src, d_k]
                             │
                             └────> Projeksiyon (W_V) ──────────> Değer (V)   [B, H, S_src, d_v]
                                                                        │
                                                                   [ Ağırlıklı Toplam ]
                                                                        │
                                                                        ▼
                                                             Hizalanmış Çıktı [B, S_tgt, d_model]
```

### Dinamik Hizalama (Dynamic Alignment):
Cross-Attention matrisinin satırları hedef kelimeleri ($S_{\text{tgt}}$), sütunları ise kaynak kelimeleri ($S_{\text{src}}$) temsil eder. Bu matris, geleneksel istatistiksel makine çevirisindeki (SMT) kelime hizalama tablolarının (alignment tables) dinamik, sürekli ve türevlenebilir bir eşdeğeridir.

---

## 3. Eğitim Sırasında Paralellik: Öğretmen Zorlaması (Teacher Forcing)

Decoder doğası gereği ardışıl kelime üretse de, **eğitim aşamasında döngüye girmeden tek bir matris adımıyla $O(1)$ sürede eğitilir**.

Bunu sağlayan teknik **Teacher Forcing** ve **Causal Masking (Nedensellik Maskesi)** kombinasyonudur:
* Decoder'a hedef cümlenin tamamı bir kaydırılmış olarak (`<BOS>, y_1, y_2, \dots, y_{T-1}`) tek seferde verilir.
* Ancak modelin gelecekteki kelimeleri ($y_i$ için $y_{i+1}, \dots$) görerek kopya çekmesini engellemek için dikkat matrisinin üst üçgeni $-\infty$ ile maskelenir:

$$M_{\text{causal}} = \begin{pmatrix} 
0 & -\infty & -\infty & -\infty \\ 
0 & 0 & -\infty & -\infty \\ 
0 & 0 & 0 & -\infty \\ 
0 & 0 & 0 & 0 
\end{pmatrix}$$

Böylece $y_1$ yalnızca `<BOS>`'u, $y_2$ ise yalnızca `<BOS>` ve $y_1$'i görerek bir sonraki kelimeyi eş zamanlı olarak tahmin eder.

---

## 4. Çıkarım Aşaması: Otoregresif Üretim Döngüsü (Inference)

Eğitimden farklı olarak, çıkarım sırasında gelecekteki kelimeler henüz bilinmemektedir. Bu nedenle üretim adım adım döngüyle gerçekleşir:

```python
# Açgözlü Çözümleme (Greedy Decoding) Mantığı:
bellek = model.encode(kaynak_cumle)
hedef = [BOS_TOKEN]

for adim in range(maks_uzunluk):
    cikti = model.decode(hedef, bellek)
    sonraki_token = cikti[:, -1].argmax(dim=-1)
    hedef.append(sonraki_token)
    if sonraki_token == EOS_TOKEN:
        break
```

---

## 5. Modern Mimari Aileleri: Encoder-Only vs. Decoder-Only vs. Encoder-Decoder

| Mimari Türü | Temsilciler | Dikkat Yapısı | En Uygun Görevler |
| :--- | :--- | :--- | :--- |
| **Encoder-Only** | BERT, RoBERTa, DeBERTa | Çift Yönlü (Bidirectional) Tam Dikkat | Sınıflandırma, Adlandırılmış Varlık Tanıma (NER), Cümle Gömme |
| **Decoder-Only** | GPT-3/4, LLaMA, Mistral, Gemma | Nedensel Tek Yönlü (Causal Masked) | Metin Üretimi, Diyalog, Kod Yazımı, Akıl Yürütme |
| **Encoder-Decoder** | Orijinal Transformer, T5, BART | Çift Yönlü Encoder + Maskeli Decoder | Makine Çevirisi, Metin Özetleme, Soru-Cevap Dönüşümü |
