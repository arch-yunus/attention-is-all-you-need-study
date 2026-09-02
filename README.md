<div align="center">

# attention-is-all-you-need-study

![Attention Is All You Need Banner](assets/banner.jpg)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Tests: Passing](https://img.shields.io/badge/Tests-11%20Passing-brightgreen.svg)](tests/)

</div>

> *"Dizi dönüşüm modellerinin hâkimi olan yaklaşımlar; bir kodlayıcı (encoder) ve kod çözücü (decoder) içeren karmaşık tekrarlayan (recurrent) veya evrişimli (convolutional) sinir ağlarına dayanmaktadır. En yüksek başarıma sahip modeller dahi kodlayıcı ile kod çözücüyü bir dikkat mekanizması üzerinden birbirine bağlar. Biz bu çalışmada; tekrarlı döngüleri ve evrişimleri tamamen bir kenara bırakan, bütünüyle dikkat mekanizmalarına dayalı yeni ve yalın bir ağ mimarisi olan Transformer'ı öneriyoruz."*  
> — **Ashish Vaswani vd. (Google Brain & Google Research, 2017)**

---

Bu depo; modern Büyük Dil Modellerinin (LLM), üretken yapay zekanın ve temel model mimarilerinin miladı sayılan **"Attention Is All You Need"** (Vaswani et al., 2017) makalesinin satır satır kavramsal tahlili, matematiksel ispatları, tensör dönüşüm mekanikleri ve sıfırdan referans implementasyonunu sunan kapsamlı bir açık kaynak Türkçe araştırma ve eğitim kılavuzudur.

---

## 1. Kronoloji ve Paradigma Kırılması: RNN Dünyasından Matris Paralelliğine

Makalenin yayınlandığı Haziran 2017 öncesinde dizi modelleme sahası ardışıl (sequential) modellerin mutlak kontrolü altındaydı. Girdi dizisi zaman ekseninde adım adım işleniyor, her yeni gizli durum bir önceki zamana kilitleniyordu.

```
RNN / LSTM Darboğazı:
t_0 ---> [ Hücre ] ---> h_0
             ↓
t_1 ---> [ Hücre ] ---> h_1   (h_0 hesaplanmadan t_1 işlenemez: O(N) zaman karmaşıklığı)
             ↓
t_2 ---> [ Hücre ] ---> h_2

Transformer Paradigması:
[ t_0, t_1, t_2 ] ---> [ Q, K, V Projeksiyonları ] ---> [ Dikkat Matrisi ] ---> [ Çıktı ]
(Tüm tokenlar tek bir matris çarpımıyla anında, eş zamanlı ve O(1) yol uzunluğunda etkileşir)
```

> *"Doğası gereği ardışıl olan bu yapı, eğitim örnekleri içerisindeki paralelleştirmeyi imkânsız kılar. Bellek kısıtları örnekler arası toplu işlemeyi (batching) sınırlandırdığından, bu durum uzun dizi uzunluklarında kritik bir probleme dönüşür."*  
> — **Makaleden Alıntı: Bölüm 1 (Giriş)**

### Neden Eski Mimari Tıkandı?

* **Ardışıl Hesaplama Darboğazı:** $h_t = f(h_{t-1}, x_t)$ yapısı bir zincirdir. GPU'ların binlerce çekirdekle sunduğu devasa tensör paralelleştirme kabiliyeti bu bağımlılık yüzünden atıl kalıyordu.
* **Bellek Uçurumu ve Bilgi Kaybı (Vanishing Information):** Cümlenin başında yer alan bir özne ile 100 kelime sonra gelen yüklem arasındaki etkileşim, aradaki 100 gizli durum matrisinden çarparak geçmek zorundaydı. LSTM kapıları bu sorunu hafifletse de bilgi kaybı matematiksel bir sınır olarak varlığını sürdürdü.
* **Hesaplama Yolu Uzunluğu (Path Length):** İki sinyal arasındaki etkileşimin yol uzunluğu RNN'lerde $O(N)$ iken, Transformer mimarisinde iki token arasındaki mesafe daima $O(1)$'dir.

---

## 2. Alanın Öncüleri ve Mimarlar Ne Dedi?

Transformer mimarisinin yarattığı etki salt bir model önerisi olmanın ötesine geçerek tüm hesaplamalı bilimlerin yönünü değiştirdi:

> *"Transformer'lar olağanüstü derecede genel bir mimari olduğunu kanıtladı. Yalnızca bu 'self-attention' (öz-dikkat) fikrinin metin, görsel, ses, video işleyebildiği ve hatta robot kontrolü yapabildiği ortaya çıktı. Bu mimari, derin öğrenme için bugüne kadar bulabildiğimiz evrensel bir hesaplama temeline en yakın şeydir."*  
> — **Andrej Karpathy** (Eski Tesla AI Direktörü, OpenAI Kurucu Ortağı)

> *"Transformer, son on yılın en başarılı yapay sinir ağı mimarisidir. Sahayı kasıp kavurdu; çünkü hem RNN'lere kıyasla temelde çok daha yüksek paralelleştirilebilirliğe sahipti hem de veri ve hesaplama gücü arttıkça olağanüstü bir ölçeklenme gösterdi."*  
> — **Yann LeCun** (Meta Baş Yapay Zeka Bilim İnsanı, Turing Ödülü Sahibi)

> *"Makalenin başlığı adeta bir başkaldırı bildirisiydi: 'Döngülere ihtiyacınız yok, evrişimlere ihtiyacınız yok; gerçekten tek ihtiyacınız olan şey dikkat mekanizmasıdır.' O dönem için kulağa inanılmaz cüretkâr, neredeyse kibirli geliyordu; fakat matematik ve deneysel ölçeklenme sonuçları bu iddiayı sonuna kadar haklı çıkardı."*  
> — **Illia Polosukhin** (Makalenin ortak yazarı, NEAR Protocol Kurucu Ortağı)

> *"'Attention Is All You Need' makalesini kaleme aldığımızda, makine çevirisi için LSTM'lerden çok daha iyi olduğunu biliyorduk; ancak hiçbirimiz bilgisayarlı görüyü, protein katlanmasını, kod üretimini tamamen dönüştüreceğini ve modern LLM çağını doğrudan başlatacağını tahmin etmemiştik."*  
> — **Aidan Gomez** (Makalenin ortak yazarı, Cohere Kurucusu & CEO'su)

> *"70 yıllık yapay zeka araştırmalarından çıkarılacak en büyük ders; hesaplama gücünden sonuna kadar yararlanan genel yöntemlerin eninde sonunda açık ara en etkili yöntemler olduğudur... Acı ders şudur: Kendi düşünme biçimimizi modellerin içine inşa etmeye çalışmak uzun vadede hiçbir işe yaramaz."*  
> — **Rich Sutton** (*The Bitter Lesson / Acı Ders*, 2019)  
> *(Transformer mimarisi, insan zihninin ardışıl okuma varsayımlarını bir kenara bırakıp donanımın matris hesaplama gücüne doğrudan bağlandığı için bu tezin en somut kanıtıdır.)*

---

## 3. Matematiksel Çekirdek: Scaled Dot-Product Attention

> *"Bir dikkat fonksiyonu; bir sorguyu (query) ve bir dizi anahtar-değer (key-value) çiftini bir çıktıya eşlemek olarak tanımlanabilir. Burada sorgu, anahtarlar, değerler ve çıktının tamamı birer vektördür."*  
> — **Makaleden Alıntı: Bölüm 3.2 (Dikkat)**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

```
   Sorgu (Q)       Anahtar (K)
       \               /
        \             /
         [ Q × K^T ]   -----------------> [B, H, S, S] Ham Benzerlik Matrisi
              |
              |
        [ / sqrt(d_k) ] ----------------> Gradyan doyumunu önleyen kritik ölçekleme
              |
              |
        [ Maske (Ops.) ] ---------------> Causal ya da Dolgu Maskesi (-inf ekleme)
              |
              |
          [ Softmax ] ------------------> Olasılık Dağılımına Dönüştürme (Satır toplamı = 1)
              |
              \        Değer (V)
               \      /
               [ Matmul ] --------------> Dikkat Ağırlıklı Temsil
                   |
                 Çıktı
```

### $\sqrt{d_k}$ Bölümünün Matematiksel İspatı ve Gradyan Doyumu (Saturation)

> *"$d_k$'nın büyük değerlerinde iç çarpım sonuçlarının genlik olarak çok büyüdüğünü, bunun da softmax fonksiyonunu aşırı derecede küçük gradyanlara sahip bölgelere ittiğini tahmin ediyoruz. Bu etkiyi ortadan kaldırmak için iç çarpımları $\frac{1}{\sqrt{d_k}}$ ile ölçekliyoruz."*  
> — **Makaleden Alıntı: Bölüm 3.2.1**

* $q$ ve $k$ bileşenlerinin ortalaması 0, varyansı 1 ($\mathbb{E}[q_i] = 0, \text{Var}(q_i) = 1$) olan bağımsız rassal değişkenler olduğunu varsayalım.
* İki vektörün iç çarpımı:

$$S = \sum_{i=1}^{d_k} q_i k_i$$

* Toplamın beklenen değeri ve varyansı:

$$\mathbb{E}[S] = \sum_{i=1}^{d_k} \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$$

$$\text{Var}(S) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k} 1 = d_k$$

* $d_k = 64$ veya $128$ gibi boyutlarda iç çarpım sonuçlarının varyansı $d_k$'ya eşittir; yani değerler $\pm \sqrt{d_k}$ bandında dağılır.
* Softmax fonksiyonuna giren girdiler mutlak değer olarak büyüdüğünde çıktı en büyük eleman için 1'e, diğerleri için 0'a kilitlenir.
* Softmax türevi $\frac{\partial S_i}{\partial z_j} = S_i(\delta_{ij} - S_j)$ ifadesi gereği $S_i \approx 1$ veya $S_i \approx 0$ durumunda türev sıfıra yaklaşır (**gradyan sönümlenmesi**). $\sqrt{d_k}$ ile bölmek varyansı tekrar $1.0$ seviyesine çekerek geriye yayılımın (backpropagation) kararlı çalışmasını sağlar.

---

## 4. Multi-Head Attention (Çok Başlı Temsil Mekaniği)

<div align="center">
  <img src="assets/multi_head_attention.jpg" alt="Multi-Head Attention Mekanizması" width="85%" />
</div>

> *"Tek bir dikkat fonksiyonunu $d_{model}$ boyutundaki sorgular, anahtarlar ve değerlerle çalıştırmak yerine; sorgu, anahtar ve değerleri $h$ kez farklı ve öğrenilmiş doğrusal projeksiyonlarla doğrusal olarak yansıtmanın faydalı olduğunu gördük... Çok başlı dikkat, modelin farklı konumlardaki farklı temsil alt uzaylarındaki bilgilere aynı anda odaklanabilmesini sağlar."*  
> — **Makaleden Alıntı: Bölüm 3.2.2**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

* **Tek Baş Kısıtı:** Tek bir dikkat başı, kelimelerin ilişkilerini tek bir ağırlıklı ortalama matrisine zorlar.
* **Alt Uzay Dağılımı:** Çoklu başlar sayesinde:
  * **Head 1:** Gramatikal yapıları ve özne-yüklem bağlarını yakalar.
  * **Head 2:** Zamir referanslarını (coreference resolution) çözümler.
  * **Head 3:** Doğrudan yanındaki kelimelere (yerel bağlama) odaklanır.
  * **Head 4:** Cümle başındaki bağlaçlar ile sonundaki noktalama arasındaki uzun menzilli köprüleri izler.

---

## 5. Sinüzoidal Pozisyonel Kodlama (Positional Encoding)

<div align="center">
  <img src="assets/positional_encoding.jpg" alt="Sinüzoidal Pozisyonel Kodlama" width="85%" />
</div>

> *"Modelimiz yineleme (recurrence) ve evrişim (convolution) içermediğinden, dizilimdeki token'ların sırasından faydalanabilmesi için dizideki token'ların göreli ya da mutlak konumlarına dair bazı bilgileri modele enjekte etmemiz gerekir."*  
> — **Makaleden Alıntı: Bölüm 3.5**

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

* **Neden Ekleme Yapılır?** Vektör boyutunu ikiye katlamadan temsil alanını korumak için token embedding ile pozisyon vektörü doğrudan toplanır ($X + PE$). Model yüksek boyutlu uzayda token anlamı ile pozisyon frekansını birbirinden ayrıştırmayı öğrenir.
* **Doğrusal Dönüşüm Kabiliyeti:** Trigonometrik toplam-fark kurallarına göre herhangi sabit bir $k$ adımı için $PE_{pos+k}$, $PE_{pos}$ vektörünün sabit bir rotasyon matrisi $M_k$ ile çarpılmasıyla ($\mathbf{PE}_{pos+k} = M_k \cdot \mathbf{PE}_{pos}$) türetilebilir. Bu formül modelin mutlak konumların yanında token'lar arası **göreli mesafeleri** de kolayca öğrenmesini sağlar.

---

## 6. Maskeleme Mekanizmaları

| Maske Türü | Kullanıldığı Konum | Amaç | Yöntem |
| :--- | :--- | :--- | :--- |
| **Padding Mask** | Encoder & Decoder | Batched girdilerdeki dolgu (`<PAD>`) token'larının hesaba katılmasını engellemek | Padding indekslerine denk gelen pozisyonlara Softmax öncesinde `-inf` (veya $-1e9$) atanır. |
| **Causal / Look-Ahead Mask** | Sadece Decoder (Öz-Dikkat) | Modelin gelecekteki token'ları görerek "kopya çekmesini" önlemek (Oto-regresif üretim) | Dikkat matrisinin üst üçgeni (upper triangular matrix) `-inf` ile sıfırlanır. |

> *"Kod çözücü katmanındaki alt katmanı, konumların sonraki konumlara dikkat yöneltmesini engelleyecek şekilde düzenledik... Bu maskeleme, çıktı gömmelerinin bir konum kaydırılmış olmasıyla birleştiğinde; $i$ konumu için yapılan tahminlerin yalnızca $i$'den küçük konumlardaki bilinen çıktılara bağlı olmasını güvenceye alır."*  
> — **Makaleden Alıntı: Bölüm 3.1**

---

## 7. Model Mimarisi ve Depo Yapısı

<div align="center">
  <img src="assets/transformer_architecture.jpg" alt="Transformer Mimarisi" width="85%" />
</div>

```text
attention-is-all-you-need-study/
├── assets/                                # Görseller, Banner ve Bilimsel Çizimler
│   ├── banner.jpg                         # Depo ana başlık afişi
│   ├── transformer_architecture.jpg       # Tam Transformer mimarisi şeması
│   ├── multi_head_attention.jpg           # Çok başlı dikkat 3D diyagramı
│   ├── positional_encoding.jpg            # Pozisyonel frekans dalgaları görseli
│   ├── positional_encoding_heatmap.png    # Matplotlib PE ısı haritası ve benzerlik matrisi
│   └── noam_lr_curve.png                  # Noam öğrenme oranı ısınma eğrisi grafiği
├── docs/                                  # Ayrıntılı Akademik Dokümantasyon
│   ├── 00_tarihsel_baglam.md             # LSTM/RNN kısıtları ve donanım darboğazları
│   ├── 01_matematiksel_temeller.md       # İç çarpım varyansı, türevler ve softmax doyumu
│   ├── 02_cok_basli_dikkat_geometrisi.md # Alt uzay projeksiyonları ve tensör katlama işlemleri
│   ├── 03_pozisyonel_kodlama.md          # Dalga boyu spektrumu ve göreli lineer dönüşüm ispatı
│   ├── 04_encoder_decoder_koprusu.md     # Cross-attention mantığı (Q Decoder'dan, K-V Encoder'dan)
│   └── 05_layer_norm_ve_residual.md      # Pre-LN vs Post-LN mimari analizleri
├── notebooks/                             # İnteraktif Jupyter Defterleri
│   ├── 01_adim_adim_tensor_boyutlari.ipynb # Her katmanda tensör boyutlarının izlenmesi
│   ├── 02_pozisyonel_dalga_boylari.ipynb   # 10000 tabanlı sin/cos dalga frekans haritaları
│   └── 03_dikkat_haritalari.ipynb          # Çoklu başların dikkat haritalarının görselleştirilmesi
├── src/                                   # Referans PyTorch İmplementasyonu
│   ├── __init__.py
│   ├── scaled_dot_product.py             # Saf tensör operasyonlu çekirdek dikkat mekanizması
│   ├── multi_head_attention.py           # Paralelleştirilmiş projeksiyon matrisleri
│   ├── positional_encoding.py            # Analitik sinüzoidal pozisyon kodlayıcı
│   ├── feed_forward.py                   # Position-wise iki katmanlı MLP bloğu
│   ├── residual_norm.py                  # Add & LayerNorm (Post-LN ve Pre-LN)
│   ├── encoder.py                        # N x EncoderLayer mimarisi
│   ├── decoder.py                        # N x DecoderLayer mimarisi
│   ├── transformer.py                    # Uçtan uca saf referans modeli
│   ├── masks.py                          # Causal ve Padding maske oluşturucuları
│   ├── optimizer.py                      # Noam Learning Rate Scheduler (Bölüm 5.3)
│   └── label_smoothing.py                # Label Smoothing Loss (Bölüm 5.4)
├── papers/
│   └── 1706.03762v7.pdf                  # Orijinal arXiv araştırma makalesi
├── tests/                                 # Kapsamlı Pytest Doğrulama Suiti
│   ├── test_shapes.py                    # Katmanlar arası tensör boyut bütünlüğü testleri
│   ├── test_causal_mask.py               # Gelecek sızıntısı (leakage) doğrulama testleri
│   └── test_components.py               # Noam LR, Label Smoothing ve Pre-LN testleri
├── example_training.py                    # Uçtan uca sentetik eğitim ve greedy decoding demosu
├── requirements.txt                       # Gerekli Python kütüphaneleri
├── pytest.ini                             # Test konfigürasyonu
├── LICENSE                                # MIT Lisansı
└── README.md
```

---

## 8. Hızlı Başlangıç ve Çalıştırma (Quickstart)

### Gereksinimlerin Yüklenmesi
```bash
pip install -r requirements.txt
```

### Test Suitini Çalıştırma
Tüm modüllerin tensör boyutlarını, maskeleme mantığını ve sızıntı testlerini doğrulamak için:
```bash
pytest tests/ -v
```

### Örnek Eğitimi Başlatma
Sentetik bir dizi öğrenme görevi üzerinde Noam scheduler ve Label Smoothing kullanarak küçük bir Transformer modelini eğitmek ve otoregresif çıkarımını izlemek için:
```bash
python example_training.py
```

### İnteraktif Görselleştirmeleri İnceleme
```bash
jupyter notebook notebooks/
```

---

## 9. Akademik Referans

```bibtex
@inproceedings{vaswani2017attention,
  author    = {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and 
               Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
  title     = {Attention Is All You Need},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS 2017)},
  volume    = {30},
  year      = {2017},
  url       = {https://arxiv.org/abs/1706.03762}
}
```