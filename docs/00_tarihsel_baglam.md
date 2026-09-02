# 00. Tarihsel Bağlam: RNN Dünyasından Matris Paralelliğine

> *"Doğası gereği ardışıl olan bu yapı, eğitim örnekleri içerisindeki paralelleştirmeyi imkânsız kılar. Bellek kısıtları örnekler arası toplu işlemeyi (batching) sınırlandırdığından, bu durum uzun dizi uzunluklarında kritik bir probleme dönüşür."*  
> — **Vaswani et al., 2017 (Bölüm 1)**

---

## 1. Giriş: Dizi Modellemenin Evrimi

2017 yılından önce Doğal Dil İşleme (NLP) ve dizi dönüşüm (sequence transduction) sahasında tartışmasız bir hegemonya vardı: **Tekrarlayan Sinir Ağları (Recurrent Neural Networks - RNN)** ve bunların kapılı türevleri olan **LSTM (Long Short-Term Memory)** ve **GRU (Gated Recurrent Unit)**.

Metinler, ses dalgaları ve zaman serileri gibi veriler doğası gereği ardışıl (sequential) kabul ediliyordu: bir kelime ancak kendisinden önceki kelimeler okunduktan sonra anlam kazanırdı. Ancak bu sezgisel kabul, derin öğrenmenin donanım devrimiyle (GPU kümleri ve tensör çekirdekleri) doğrudan çatıştı.

---

## 2. RNN ve LSTM Mimarisinin Temel Darboğazları

### 2.1 Ardışıl Hesaplama Bağımlılığı (Sequential Computation Bottleneck)

Standart bir RNN hücresinin gizli durum (hidden state) güncellemesi şu formüle dayanır:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$

Bu denklemde $t$ anındaki durumun ($h_t$) hesaplanabilmesi için $h_{t-1}$ durumunun bellekte hazır bulunması şarttır.

```
Zaman Ekseni (t):
x_1 ──> [ RNN ] ──> h_1
           │
           ▼ (beklemek zorunda)
x_2 ──> [ RNN ] ──> h_2
           │
           ▼ (beklemek zorunda)
x_3 ──> [ RNN ] ──> h_3
```

* **Donanım Atıllığı:** Modern grafik işlemciler (GPU), binlerce çekirdeğiyle devasa matrisleri tek bir saat döngüsünde paralel olarak çarpmak üzere tasarlanmıştır. Ancak $t_1 \to t_2 \to \dots \to t_N$ bağımlılığı GPU'yu zorunlu bir bekleme döngüsüne sokar. $N$ uzunluğundaki bir dizinin işlenmesi en az $O(N)$ zaman adımı gerektirir.

### 2.2 Bellek Uçurumu ve Bilgi Kaybı (Vanishing Gradient & Information Bottleneck)

Encoder-Decoder RNN mimarisinde (Sutskever et al., 2014; Cho et al., 2014), kaynak cümlenin tüm anlamı tek bir sabit boyutlu vektöre ($h_T$) sıkıştırılmak zorundaydı:

$$x_1, x_2, \dots, x_T \xrightarrow{\text{RNN Encoder}} h_T \xrightarrow{\text{RNN Decoder}} y_1, y_2, \dots, y_{T'}$$

Cümle uzunluğu arttıkça (özellikle $T > 20$):
1. **Gradyan Kaybı:** Geriye yayılım (BPTT - Backpropagation Through Time) sırasında gradyanlar geriye doğru $W_{hh}^T$ matrisiyle defalarca çarpılır. Özdeğerler $< 1$ ise gradyanlar sönümlenir (vanishing gradient), $> 1$ ise patlar (exploding gradient).
2. **Sabit Boyutlu Şişe Boğazı:** 100 kelimelik zengin bir paragrafın tüm anlamsal inceliklerinin 512 veya 1024 boyutlu tek bir vektöre sığdırılması imkansız bir bilgi kaybı yaratır.

### 2.3 Hesaplama Yolu Uzunluğu (Maximum Path Length)

Bir modelde iki uzak bilgi parçasının birbiriyle etkileşime girebilmesi için sinyalin kat etmesi gereken katman/zaman adımı sayısına **hesaplama yolu uzunluğu** denir.

| Mimari Türü | İki Konum Arası Yol Uzunluğu | Adım Başı İşlem Karmaşıklığı | Asgari Sıralı İşlem Sayısı |
| :--- | :---: | :---: | :---: |
| **Recurrent (RNN/LSTM)** | $O(N)$ | $O(N \cdot d^2)$ | $O(N)$ |
| **Convolutional (ByteNet/ConvS2S)** | $O(\log_k(N))$ veya $O(N/k)$ | $O(k \cdot N \cdot d^2)$ | $O(1)$ |
| **Self-Attention (Transformer)** | $\mathbf{O(1)}$ | $\mathbf{O(N^2 \cdot d)}$ | $\mathbf{O(1)}$ |

* RNN'lerde 1. kelime ile 100. kelime arasındaki yol uzunluğu **100** adımdır.
* ConvS2S (Gehring et al., 2017) ve ByteNet (Kalchbrenner et al., 2016) gibi evrişimli mimariler genişleyen pencerelerle (dilated convolutions) bu mesafeyi $O(\log N)$ seviyesine indirse de tam etkileşim için derin hiyerarşi gerekir.
* **Self-Attention** mimarisinde ise her token diğer tüm token'larla **doğrudan** iç çarpım hesaplar; mesafe her zaman **$O(1)$**'dir.

---

## 3. Bahdanau Dikkat Mekanizması: İlk Kıvılcım (2014)

Transformer'a giden yolda ilk kritik kırılma Bahdanau et al. (2014) tarafından önerilen "Additive Attention" ile yaşandı. Amaç, sabit boyutlu $h_T$ darboğazını kırmaktı.

Decoder, $i$. hedef kelimeyi üretirken encoder'ın tüm gizli durumlarına ($h_1, \dots, h_{T}$) bakarak dinamik bir bağlam vektörü $c_i$ üretir:

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^T \exp(e_{ik})}, \quad e_{ij} = v_a^T \tanh(W_a s_{i-1} + U_a h_j)$$

$$c_i = \sum_{j=1}^T \alpha_{ij} h_j$$

Burada dikkat mekanizması RNN'i destekleyen bir "koltuk değneği" vazifesi görüyordu.

---

## 4. "Attention Is All You Need" Devrimi: Radikal Kopuş

Ashish Vaswani ve Google Brain / Research ekibi şu cesur soruyu sordu:

> *"Eğer dikkat mekanizması uzak kelimeler arasındaki ilişkiyi bu kadar kusursuz çözüyorsa, RNN döngülerine ve CNN filtrelerine gerçekten ihtiyacımız var mı?"*

Cevap makalenin başlığındaydı: **"Attention Is All You Need"** (İhtiyacınız Olan Tek Şey Dikkat).

Transformer mimarisi:
1. **Tekrarlamayı (Recurrence) tamamen attı:** Zaman bağımlılığı ortadan kalktı, tüm dizi tek bir devasa tensör olarak GPU'ya beslendi.
2. **Evrişimi (Convolution) tamamen attı:** Yerel pencere kısıtlamaları kaldırıldı; her token tüm diziye sınırsız erişim kazandı.
3. **Pozisyon Bilgisini Analitik Olarak Enjekte Etti:** Ardışıllık ortadan kalktığı için kelimelerin sırası sinüzoidal pozisyonel kodlama ile tensörlere dahil edildi.
4. **Matris Çarpım Paralelizmini Maksimize Etti:** BLAS ve cuBLAS kütüphanelerinin en optimize çalıştığı $GEMM$ (General Matrix Multiply) işlemlerine dayandı.

Sonuç: Eğitim sürelerinde onlarca kat hızlanma, çok daha derin modellerin eğitilebilmesi ve günümüzdeki GPT, BERT, Gemini ve Claude gibi milyarlarca parametreli Büyük Dil Modellerinin (LLM) temeli.
