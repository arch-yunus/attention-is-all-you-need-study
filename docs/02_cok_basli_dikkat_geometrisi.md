# 02. Çok Başlı Dikkat Geometrisi: Alt Uzay Projeksiyonları ve Tensör Katlama

> *"Tek bir dikkat fonksiyonunu $d_{model}$ boyutundaki sorgular, anahtarlar ve değerlerle çalıştırmak yerine; sorgu, anahtar ve değerleri $h$ kez farklı ve öğrenilmiş doğrusal projeksiyonlarla yansıtmanın faydalı olduğunu gördük... Çok başlı dikkat, modelin farklı konumlardaki farklı temsil alt uzaylarındaki bilgilere aynı anda odaklanabilmesini sağlar."*  
> — **Vaswani et al., 2017 (Bölüm 3.2.2)**

---

<div align="center">
  <img src="../assets/multi_head_attention.jpg" alt="Multi-Head Attention Mimarisi" width="85%" />
</div>

---

## 1. Neden Tek Bir Baş (Single Head) Yetersizdir?

Tek başlı (single-head) bir dikkat katmanında, iki kelime arasındaki benzerlik tek bir skalar ağırlıkla ($0$ ile $1$ arasında) özetlenir. 

Örneğin şu cümleyi ele alalım:
> *"Banka, nehir kıyısındaki yeni şubesini açtı."*

"Banka" kelimesi aynı anda birden fazla farklı anlamsal ilişki ağı içerisindedir:
1. **Sözdizimsel (Syntactic) İlişki:** Cümlenin öznesidir, "açtı" yüklemiyle doğrudan bağlıdır.
2. **Semantik / Bağlamsal İlişki:** "şube" kelimesi finansal kurum anlamına işaret eder.
3. **Mekânsal İlişki:** "kıyı" kelimesi coğrafi bağlamı temsil eder.

Tek bir dikkat başı kullanılırsa, bu baş tüm bu zıt ilişkilerin tek bir ağırlıklı ortalamasını (weighted average) almak zorunda kalır. Farklı boyutlardaki zengin ilişkiler birbirini nötrler veya bulanıklaştırır.

---

## 2. Çoklu Baş (Multi-Head) Mekanizması ve Alt Uzay Ayrışımı

Çok başlı dikkat, modelin $d_{model}$ boyutlu temsil uzayını $h$ adet daha küçük **temsil alt uzayına (representation subspace)** böler:

$$d_k = d_v = \frac{d_{model}}{h}$$

Orijinal makalede:
* $d_{model} = 512$
* $h = 8$
* $d_k = 512 / 8 = 64$

Her bir baş ($i = 1, \dots, h$) kendi özel projeksiyon matrislerine sahiptir:
* $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$
* $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$
* $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

Burada $W^O \in \mathbb{R}^{(h \cdot d_v) \times d_{model}}$ matrisi tüm başlardan gelen çıktıları harmanlayıp tekrar ana model boyutuna projeksiyonlar.

---

## 3. Başların Tipik Uzmanlaşma Alanları

Ampirik araştırmalar (Clark et al., 2019 - "What Does BERT Look At?"), eğitilmiş Transformer modellerinde başların belirli dilbilgisel rolleri kendiliğinden öğrendiğini kanıtlamıştır:

```
┌──────────────┐ ───> Baş 1: Doğrudan bir önceki token'a odaklanır (Yerel N-gram)
│              │ ───> Baş 2: Doğrudan bir sonraki token'a odaklanır (İleriye bakış)
│  Girdi Dizi  │ ───> Baş 3: Zamir referanslarını çözümler ("o" -> "Ali")
│ [B, S, 512]  │ ───> Baş 4: Özne-yüklem bağını kurar (Cümle başı <-> Cümle sonu)
│              │ ───> Baş 5: Noktalama işaretlerine odaklanır (Cümle sınırları)
└──────────────┘ ───> Baş 6-8: Nadir kelimeler ve özel bağlamsal semantik köprüler
```

---

## 4. Adım Adım Tensör Boyutları ve Katlama Mekaniği

Uygulamada $h$ adet bağımsız lineer katman tanımlamak yerine, tüm başlar tek bir büyük matrisle tek seferde paralel olarak hesaplanır.

### Tensör Dönüşüm Haritası:

| Adım | İşlem | Tensör Şekli (Shape) | Açıklama |
| :---: | :--- | :--- | :--- |
| **0** | Girdi | `[B, S, 512]` | $B$: Batch boyutu, $S$: Dizi uzunluğu |
| **1** | $W_Q, W_K, W_V$ Projeksiyonları | `[B, S, 512]` | $512 \to 512$ doğrusal dönüşüm |
| **2** | Başlara Ayırma (`view`) | `[B, S, 8, 64]` | Son boyut $h \times d_k$ şeklinde bölünür |
| **3** | Eksen Değişimi (`transpose(1, 2)`) | `[B, 8, S, 64]` | Baş boyutu öne alınarak $B \times 8$ paralel matris elde edilir |
| **4** | $Q \times K^T$ İç Çarpımı | `[B, 8, S, S]` | Her baş için $S \times S$ ham benzerlik haritası |
| **5** | $/ \sqrt{d_k} + \text{Maske} + \text{Softmax}$ | `[B, 8, S, S]` | Normalize edilmiş dikkat ağırlıkları matrisi |
| **6** | $\text{Ağırlıklar} \times V$ Çarpımı | `[B, 8, S, 64]` | Değer vektörlerinin dikkat ağırlıklı toplamı |
| **7** | Transpozisyon (`transpose(1, 2)`) | `[B, S, 8, 64]` | Dizi ve baş boyutları eski sırasına getirilir |
| **8** | Başları Birleştirme (`view/contiguous`) | `[B, S, 512]` | 8 baş yan yana yapıştırılır ($8 \times 64 = 512$) |
| **9** | Çıkış Projeksiyonu ($W^O$) | `[B, S, 512]` | $512 \to 512$ harmanlama dönüşümü |

---

## 5. Hesaplama Karmaşıklığı: Multi-Head Bedava mı?

Çok başlı dikkatin en zarif yönlerinden biri, tek başlı tam boyutlu dikkate kıyasla **ekstra bir hesaplama maliyeti getirmemesidir**.

* **Tek Baş ($d_k = 512$):**
  * $Q K^T$: $[S, 512] \times [512, S] \implies S \times S \times 512$ işlem.
* **$h$ Baş ($d_k = 64, h = 8$):**
  * Her baş: $[S, 64] \times [64, S] \implies S \times S \times 64$ işlem.
  * $8$ başın toplamı: $8 \times (S \times S \times 64) = S \times S \times 512$ işlem.

Toplam FLOP miktarı **birebir aynıdır**, ancak model çoklu alt uzaylarda temsil gücünü katlar.
