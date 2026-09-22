# Bölüm 9: Modern Aktivasyon Fonksiyonları ve RMSNorm

> *"We find that GLU variants consistently yield better perplexity than standard activations across various Transformer architectures."*  
> — **Noam Shazeer**, *"GLU Variants Improve Transformer" (Google Research, 2020)*

---

## 1. Giriş ve Motivasyon

2017 yılında Vaswani vd. tarafından tanıtılan orijinal Transformer modelinde standart **Layer Normalization (Ba et al., 2016)** ve **ReLU** aktivasyon fonksiyonu kullanılmıştır. Ancak milyarlarca parametreli modern LLM'lere (LLaMA-3, Mistral, PaLM, Gemma, DeepSeek) geçildikçe iki kritik darboğaz ortaya çıkmıştır:

1. **LayerNorm Hesaplama ve Bellek Maliyeti**: Ortalama çıkarma ve varyans hesaplama, GPU bellek bant genişliğini (Memory Bandwidth) zorlar.
2. **ReLU'nun Düşük Temsil Gücü**: Negatif değerlerde sıfır gradyan vermesi ("dying ReLU") ve doğrusal olmayan temsil kapasitesinin sınırlı kalması.

Bu bölümde modern dil modellerinin omurgasını oluşturan **RMSNorm** ve **SwiGLU** mimarilerini derinlemesine inceliyoruz.

---

## 2. Kök Ortalama Kare Normalizasyonu (RMSNorm)

Zhang & Sennrich (2019) tarafından önerilen RMSNorm, standart Katman Normalizasyonu'ndaki ortalama merkezleme (mean centering) adımını iptal eder.

### Matematiksel Formülasyon

Verilen bir girdi vektörü \(x \in \mathbb{R}^d\) için:

$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$

$$\bar{a}_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i$$

Burada \(\gamma\) öğrenilebilir ölçekleme parametresidir (\(\beta\) öteleme parametresi modern modellerde genellikle kaldırılmıştır).

### LayerNorm vs. RMSNorm Karşılaştırması

| Özellik | LayerNorm (2016) | RMSNorm (2019) |
| :--- | :--- | :--- |
| **Ortalama Hesabı** | \(\mu = \frac{1}{d}\sum x_i\) (Var) | **Yok (Kaldırıldı)** |
| **Öğrenilebilir Parametreler** | \(\gamma\) (ölçek) ve \(\beta\) (kayma) | Yalnızca \(\gamma\) (ölçek) |
| **Hız / Bellek Kazancı** | Referans (1.0x) | **%10 - %50 Daha Hızlı** |
| **Kullanan Modeller** | BERT, GPT-2, GPT-3 | LLaMA-1/2/3, Mistral, Gemma, Chinchilla |

---

## 3. Swish Kapılı Doğrusal Birim (SwiGLU)

Dauphin vd. (2017) tarafından önerilen Gated Linear Unit (GLU) ailesi, Noam Shazeer (2020) tarafından Swish aktivasyonu ile birleştirilerek **SwiGLU** katmanına dönüştürülmüştür.

### Matematiksel Tanım

Standart FFN iki doğrusal dönüşümden oluşur:
$$\text{FFN}_{\text{ReLU}}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

SwiGLU ise iki ayrı projeksiyonun kapılanmış (gated) çarpımını kullanır:
$$\text{SwiGLU}(x) = \left( \text{Swish}(x W_{\text{gate}}) \otimes (x W_{\text{up}}) \right) W_{\text{down}}$$

Burada \(\text{Swish}(z) = z \cdot \sigma(z) = \text{SiLU}(z)\)'dir.

### Parametre ve Boyut Dengesi

Standart FFN'de iki ağırlık matrisi varken (\(2 \times d_{\text{model}} \times d_{\text{ff}}\)), SwiGLU'da üç ağırlık matrisi (\(W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}\)) bulunur. Parametre sayısını standart Transformer ile eşit tutmak için gizli katman boyutu:

$$d_{\text{ff}} = \left\lfloor \frac{2}{3} \times 4 d_{\text{model}} \right\rfloor = \left\lfloor \frac{8}{3} d_{\text{model}} \right\rfloor$$

olarak seçilir ve GPU tensör çekirdekleri (Tensor Cores) optimizasyonu için en yakın 256'nın katına yuvarlanır.

---

## 4. DeepNorm: Ultra-Derin Mimarilerin Kararlılığı

Wang vd. (2022) tarafından geliştirilen DeepNorm, 1000+ katmanlı modellerin (DeepNet) Post-LN mimarisinde diverjans yaşamadan eğitilmesini sağlar:

$$x_{l+1} = \text{LayerNorm}(x_l \cdot \alpha + \text{Sublayer}(x_l))$$

Encoder derinliği \(N\) ve Decoder derinliği \(M\) için:
- \(\alpha = (2N)^{1/4}\)
- \(\beta = (8N)^{-1/4}\)
