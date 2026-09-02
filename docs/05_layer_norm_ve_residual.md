# 05. Normalizasyon ve Artık Bağlantılar: Post-LN vs. Pre-LN Analizi

> *"Her bir alt katmanın çıktısı $\text{LayerNorm}(x + \text{Sublayer}(x))$ şeklindedir; burada $\text{Sublayer}(x)$ alt katmanın kendisi tarafından uygulanan fonksiyondur."*  
> — **Vaswani et al., 2017 (Bölüm 3.1)**

---

## 1. Artık Bağlantılar (Residual Connections) ve Gradyan Otoyolu

He et al. (2016) tarafından ResNet mimarisiyle derin öğrenmeye kazandırılan artık bağlantılar (skip / residual connections), derin ağların eğitilebilirliğini güvenceye alan temel omurgadır.

### Matematiksel Gradyan Akışı:
Bir alt katmanı düşünelim:
$$x_{l+1} = x_l + \mathcal{F}(x_l)$$

Kayıp fonksiyonunun ($\mathcal{L}$) bir önceki katmanın girdisine ($x_l$) göre türevi zincir kuralıyla şöyle yazılır:

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_{l+1}} \cdot \frac{\partial x_{l+1}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_{l+1}} \left( \mathbf{I} + \frac{\partial \mathcal{F}(x_l)}{\partial x_l} \right)$$

Buradaki birim matris $\mathbf{I}$, geriye yayılan gradyan sinyaline **katıksız, doğrudan ve kesintisiz bir otoyol (gradient highway)** sunar. $\mathcal{F}$ katmanının gradyanları sıfıra yaklaşsa dahi $\mathbf{I}$ terimi sayesinde gradyanlar ilk katmanlara kadar zayıflamadan iletilir.

---

## 2. Katman Normalizasyonu (Layer Normalization): Neden BatchNorm Değil?

Görüntü işlemede yaygın olan Toplu Normalizasyon (Batch Normalization - BatchNorm), istatistikleri mini-batch boyutu ($B$) boyunca hesaplar:

```
Batch Normalization (BatchNorm):           Layer Normalization (LayerNorm):
[Batch 1]  ████████                        [Batch 1]  [Feature 1 ... Feature D] ──> Ort ve Varyans
[Batch 2]  ████████                        [Batch 2]  [Feature 1 ... Feature D] ──> Ort ve Varyans
               ▲                                          ▲
               └── Mini-batch ortalaması                  └── Her örnek kendi içinde bağımsız
```

### Doğal Dil İşlemede BatchNorm'un Çöküş Nedenleri:
1. **Değişken Dizi Uzunluğu (Variable Sequence Length):** Cümleler farklı uzunluktadır. Padding token'ları toplu istatistikleri kirletir.
2. **Batch Boyutuna Bağımlılık:** Çıkarım (inference) sırasında tek bir cümle test edilirken eğitimdeki toplu istatistikler uyumsuzluk yaratır.

### LayerNorm Formülasyonu:
LayerNorm (Ba et al., 2016), normalizasyonu her örnek ve her zaman adımı için bağımsız olarak **öznitelik ekseni ($d_{model}$)** boyunca hesaplar:

$$\mu = \frac{1}{d} \sum_{i=1}^d x_i, \quad \sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$$

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

$$y = \gamma \odot \hat{x} + \beta$$

Burada $\gamma$ ve $\beta$ öğrenilebilir ölçekleme ve kaydırma parametreleridir.

---

## 3. Post-LN vs. Pre-LN Mimarisi

Orijinal makale ile günümüzün modern LLM pratikleri arasındaki en belirleyici yapısal fark normalizasyonun konumudur.

### Post-LN (Orijinal Vaswani 2017 Yaklaşımı):
Alt katmanın çıktısı $x$ ile toplanır, ardından normalizasyon uygulanır:

$$x_{l+1} = \text{LayerNorm}(x_l + \mathcal{F}(x_l))$$

```
x ───┬───────────────────────────(+) ──> LayerNorm ──> Çıktı
     │                            ▲
     └──> [ Attention / FFN ] ────┘
```

* **Avantajı:** Temsil kapasitesi teorik olarak daha yüksektir.
* **Kritik Dezavantajı:** Ağ derinleştikçe gradyan varyansı katmanlar arasında dengesizleşir. Çıkışa yakın katmanların gradyanları ilk katmanlara kıyasla çok daha büyük olur. Model warm-up (Noam scheduler) olmadan eğitilirse ilk adımlarda diverjans (loss explosion / NaN) kaçınılmazdır.

---

### Pre-LN (Modern Standart: GPT-2, LLaMA, PaLM):
Normalizasyon alt katmandan önce uygulanır, doğrudan ana artık akışa (residual stream) eklenir:

$$x_{l+1} = x_l + \mathcal{F}(\text{LayerNorm}(x_l))$$

```
x ───┬──────────────────────────────────────────(+) ──> Çıktı
     │                                          ▲
     └──> LayerNorm ──> [ Attention / FFN ] ────┘
```

* **Avantajı:** Ana artık akış saf kalır. Gradyanlar en son katmandan en ilk katmana hiçbir katman normalizasyonu tarafından sıkıştırılmadan veya bozulmadan doğrudan akar.
* **Sonuç:** Model 100+ katmana kadar hiçbir özel warm-up ayarı gerektirmeden son derece kararlı ve hızlı bir şekilde eğitilebilir.

---

## 4. Modern Evrim: RMSNorm (Root Mean Square Normalization)

Modern açık kaynak LLM'lerde (LLaMA, Mistral, Gemma), LayerNorm yerine Zhang & Sennrich (2019) tarafından önerilen **RMSNorm** tercih edilmektedir:

$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}$$

$$\bar{x} = \frac{x}{\text{RMS}(x)} \odot \gamma$$

Ortalama hesaplama ve kaydırma adımını ($\mu$ ve $\beta$) denklemden çıkararak benzer model başarımı sunarken eğitim ve çıkarım hızını %10 ila %30 oranında artırır.
