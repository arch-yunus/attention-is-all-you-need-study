# 01. Matematiksel Temeller: Scaled Dot-Product Attention ve Gradyan Analizi

> *"Büyük $d_k$ değerlerinde iç çarpımların büyüklüğü aşırı artar ve bu durum softmax fonksiyonunu son derece küçük eğimlere (gradyanlara) sahip bölgelere iter. Bu etkiyi bertaraf etmek için iç çarpımları $1/\sqrt{d_k}$ ile ölçekliyoruz."*  
> — **Vaswani et al., 2017 (Bölüm 3.2.1)**

---

## 1. Temel Dikkat Fonksiyonu

Transformer mimarisinde dikkat mekanizması, bir sorgu ($Q$) vektörünün bir dizi anahtar ($K$) vektörüyle benzerliğini hesaplayarak, bu benzerlik oranında değer ($V$) vektörlerinin ağırlıklı toplamını döndürür:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Burada tensör boyutları şöyledir:
* $Q \in \mathbb{R}^{B \times H \times S_q \times d_k}$
* $K \in \mathbb{R}^{B \times H \times S_k \times d_k}$
* $V \in \mathbb{R}^{B \times H \times S_k \times d_v}$
* Çıktı $\in \mathbb{R}^{B \times H \times S_q \times d_v}$

---

## 2. Neden $\sqrt{d_k}$ ile Bölüyoruz? Matematiksel Varyans İspatı

Bu ölçeklemenin arkasındaki matematiksel gerekçeyi adım adım kanıtlayalım.

### Varsayımlar:
1. $q \in \mathbb{R}^{d_k}$ ve $k \in \mathbb{R}^{d_k}$ iki rassal vektör olsun.
2. Bu vektörlerin her bir bileşeni ($q_i$ ve $k_i$) bağımsız ve özdeş dağılmış (i.i.d.) rassal değişkenler olsun:
   $$\mathbb{E}[q_i] = 0, \quad \text{Var}(q_i) = \sigma_q^2 = 1$$
   $$\mathbb{E}[k_i] = 0, \quad \text{Var}(k_i) = \sigma_k^2 = 1$$
3. $q_i$ ve $k_j$ değişkenleri tüm $i, j$ için birbirinden bağımsızdır: $\text{Cov}(q_i, k_j) = 0$.

### İç Çarpım İfadesi:
İki vektörün nokta çarpımı skalar bir rastgele değişken olan $S$'yi verir:

$$S = q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

### Beklenen Değer $\mathbb{E}[S]$:
Beklenen değerin lineerlik özelliği ve bağımsızlık kuralı gereği:

$$\mathbb{E}[S] = \mathbb{E}\left[\sum_{i=1}^{d_k} q_i k_i\right] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = \sum_{i=1}^{d_k} \mathbb{E}[q_i] \mathbb{E}[k_i] = \sum_{i=1}^{d_k} (0 \cdot 0) = 0$$

### Varyans $\text{Var}(S)$:
Bağımsız rastgele değişkenlerin toplamının varyansı, varyansların toplamına eşittir:

$$\text{Var}(S) = \text{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i)$$

Tek bir terimin varyans tanımı:
$$\text{Var}(q_i k_i) = \mathbb{E}[(q_i k_i)^2] - (\mathbb{E}[q_i k_i])^2$$

Bağımsızlık sebebiyle $\mathbb{E}[(q_i k_i)^2] = \mathbb{E}[q_i^2] \mathbb{E}[k_i^2]$ ve $\mathbb{E}[q_i k_i] = 0$ olduğundan:
$$\mathbb{E}[q_i^2] = \text{Var}(q_i) + (\mathbb{E}[q_i])^2 = 1 + 0 = 1$$
$$\mathbb{E}[k_i^2] = \text{Var}(k_i) + (\mathbb{E}[k_i])^2 = 1 + 0 = 1$$
$$\text{Var}(q_i k_i) = 1 \cdot 1 - 0 = 1$$

Toplam varyans:
$$\text{Var}(S) = \sum_{i=1}^{d_k} 1 = d_k$$

### Standart Sapma:
$$\sigma_S = \sqrt{\text{Var}(S)} = \sqrt{d_k}$$

Örneğin, Transformer taban modelinde $d_{model} = 512$ ve $h = 8$ olduğundan $d_k = 64$'tür.
Bu durumda iç çarpım sonuçlarının varyansı **64**, standart sapması ise **$\sqrt{64} = 8$** olur. Skalar çarpım değerleri tipik olarak $[-24, +24]$ gibi geniş bir aralığa yayılır.

---

## 3. Softmax Fonksiyonunun Gradyan Doyumu (Saturation)

Softmax fonksiyonu girdilerini olasılık dağılımına dönüştürür:

$$S_i = \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^N e^{z_j}}$$

### Softmax Jacobian Matrisi (Türevi):
Softmax'ın $z_j$'ye göre kısmi türevi:

$$\frac{\partial S_i}{\partial z_j} = \begin{cases} 
S_i (1 - S_i) & \text{eğer } i = j \\ 
-S_i S_j & \text{eğer } i \neq j 
\end{cases} = S_i (\delta_{ij} - S_j)$$

Burada $\delta_{ij}$ Kronecker delta fonksiyonudur.

### Doyum Problemi (Gradient Vanishing):
Eğer girdilerden biri ($z_m$) diğerlerinden belirgin derecede büyükse ($z_m \gg z_{k \neq m}$):
* $S_m \approx 1.0$
* Diğer tüm $S_k \approx 0.0$

Bu durumda türevleri inceleyelim:
* $i = m$ için: $\frac{\partial S_m}{\partial z_m} = S_m (1 - S_m) \approx 1.0 \cdot (1 - 1.0) = \mathbf{0.0}$
* $i \neq m$ için: $\frac{\partial S_i}{\partial z_j} = -S_i S_j \approx 0.0 \cdot S_j = \mathbf{0.0}$

**Sonuç:** Softmax çıktısı neredeyse tek bir bileşene kilitlenir ("one-hot" benzeri keskin dağılım) ve tüm türevler sıfıra yaklaşır. Geriye yayılım (backpropagation) sırasında gradyanlar bu katmanda sönümlenir, model öğrenemez hale gelir.

### $\frac{1}{\sqrt{d_k}}$ ile Ölçeklemenin Kurtarıcı Rolü:
İç çarpım skoru $z = \frac{S}{\sqrt{d_k}}$ olarak ölçeklendiğinde:

$$\text{Var}\left(\frac{S}{\sqrt{d_k}}\right) = \frac{1}{d_k} \text{Var}(S) = \frac{1}{d_k} \cdot d_k = \mathbf{1.0}$$

Varyans tekrar $1.0$ değerine çekilir! Böylece logitler dengeli bir aralıkta kalır, softmax ılımlı bir olasılık dağılımı üretir ve gradyan akışı maksimum düzeyde korunur.

---

## 4. Nokta Çarpım (Dot-Product) vs. Toplamsal (Additive) Dikkat Karşılaştırması

| Kriter | Nokta Çarpım (Dot-Product) Dikkat | Toplamsal (Additive / Bahdanau) Dikkat |
| :--- | :--- | :--- |
| **Matematiksel Formül** | $\text{softmax}(QK^T / \sqrt{d_k})V$ | $v_a^T \tanh(W_q Q + W_k K)$ |
| **GPU Verimliliği** | **Çok Yüksek** (Doğrudan matris çarpımı - GEMM) | **Düşük / Orta** (İki ayrı doğrusal katman + Tanh) |
| **Hafıza Tüketimi** | Küçük ara durumlar | Tanh öncesi genişletilmiş ara tensörler |
| **Büyük Boyutlarda Başarım** | $\sqrt{d_k}$ olmadan kötü, $\sqrt{d_k}$ ile mükemmel | Ölçekleme gerektirmez ancak yavaştır |
