# 03. Sinüzoidal Pozisyonel Kodlama: Dalga Frekansları ve Göreli Rotasyon İspatı

> *"Modelimiz yineleme veya evrişim içermediğinden; dizilimdeki token'ların sırasından faydalanabilmesi için dizideki göreli veya mutlak konumlarına dair bazı bilgileri modele enjekte etmemiz gerekir."*  
> — **Vaswani et al., 2017 (Bölüm 3.5)**

---

## 1. Permütasyon Eşdeğişirliği (Permutation Equivariance) Problemi

Self-attention mekanizması doğası gereği bir **küme (set) işlemcisidir**. Eğer bir girdideki kelimelerin sırasını rastgele karıştırırsanız, dikkat ağırlıkları da aynı şekilde yer değiştirir; ancak model hangi kelimenin önce, hangisinin sonra geldiğini anlayamaz:

$$\text{Attention}(P \cdot X) = P \cdot \text{Attention}(X)$$

Burada $P$ bir permütasyon matrisidir. Doğal dilde ise sıra hayati önem taşır:
* *"Kedi fareyi yakaladı."*
* *"Fare kediyi yakaladı."*

Bu iki cümle tamamen aynı kelimeleri içermesine rağmen taban tabana zıt anlamlara sahiptir. Sıra bilgisini modele aktarmak için Vaswani vd., **Sinüzoidal Pozisyonel Kodlama (Sinusoidal Positional Encoding)** yöntemini geliştirdi.

---

## 2. Sinüzoidal Pozisyonel Kodlama Formülasyonu

Her $pos \in [0, \text{max\_len}-1]$ konumu ve her $i \in [0, d_{model}/2 - 1]$ boyut çifti için:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{model}}}\right)$$

### Dalga Boyu ve Frekans Analizi:
Burada frekansı $\omega_i$ ile tanımlarsak:

$$\omega_i = \frac{1}{10000^{2i / d_{model}}}$$

* **İlk Boyutlar ($i = 0$):** Dalga boyu $\lambda = 2\pi \approx 6.28$ token'dır. Çok yüksek frekansta hızla salınır; yakın komşu token'ların ayrışmasını sağlar (yerel konum hassasiyeti).
* **Son Boyutlar ($i \approx d_{model}/2$):** Dalga boyu $\lambda = 2\pi \cdot 10000 \approx 62.831$ token'dır. Çok düşük frekansta yavaşça değişir; uzun menzilli genel konum bilgisini (cümlenin başı, ortası, sonu) taşır.

---

## 3. Göreli Lineer Dönüşüm İspatı (Relative Linear Transformation)

Makaledeki en derin matematiksel iddialardan biri şudur:

> *"Herhangi sabit bir $k$ adımı için, $PE_{pos+k}$ vektörü $PE_{pos}$ vektörünün doğrusal bir fonksiyonu olarak temsil edilebilir."*

Bu iddiayı trigonometrik toplam formülleriyle kanıtlayalım.

### İspat:
Trigonometrik açılımlar:
$$\sin(\alpha + \beta) = \sin(\alpha)\cos(\beta) + \cos(\alpha)\sin(\beta)$$
$$\cos(\alpha + \beta) = \cos(\alpha)\cos(\beta) - \sin(\alpha)\sin(\beta)$$

Burada $\alpha = \omega_i \cdot pos$ ve $\beta = \omega_i \cdot k$ diyelim:

$$PE_{(pos+k, 2i)} = \sin(\omega_i(pos + k)) = \sin(\omega_i pos)\cos(\omega_i k) + \cos(\omega_i pos)\sin(\omega_i k)$$

$$PE_{(pos+k, 2i+1)} = \cos(\omega_i(pos + k)) = \cos(\omega_i pos)\cos(\omega_i k) - \sin(\omega_i pos)\sin(\omega_i k)$$

Bu iki denklemi bir matris çarpımı olarak yazarsak:

$$\begin{pmatrix} PE_{(pos+k, 2i)} \\ PE_{(pos+k, 2i+1)} \end{pmatrix} = \begin{pmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{pmatrix} \begin{pmatrix} PE_{(pos, 2i)} \\ PE_{(pos, 2i+1)} \end{pmatrix}$$

Buradaki $2 \times 2$ dönüşüm matrisine $M_k^{(i)}$ rotasyon matrisi dersek:

$$M_k^{(i)} = \begin{pmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{pmatrix}$$

### Kritik Çıkarım:
$M_k^{(i)}$ matrisi **mutlak konuma ($pos$) hiçbir şekilde bağlı değildir!** Yalnızca iki kelime arasındaki göreli mesafe olan **$k$** adımına bağlıdır.

Bu sayede, öz-dikkat mekanizmasında $Q$ ve $K$ arasındaki iç çarpım hesaplanırken model mutlak konumların yanı sıra **iki token arasındaki bağıl mesafeyi ($k = pos_q - pos_k$) de doğrudan ve lineer olarak okuyabilir!**

*(Not: Bu zarif rotasyonel özellik, modern LLM'lerde kullanılan **RoPE - Rotary Position Embedding** mimarisinin de doğrudan ilham kaynağıdır).*

---

## 4. Neden Birleştirme (Concat) Değil de Toplama (Addition)?

Pozisyon vektörü token gömmesiyle birleştirilmek (`[embed; PE]`, boyut: $2 \times d_{model}$) yerine doğrudan toplanır (`embed + PE`, boyut: $d_{model}$):

$$x_{\text{input}} = \text{Embedding}(token) + PE$$

* **Boyut Tasarrufu:** Parametre sayısını ve matris boyutlarını şişirmez.
* **Yüksek Boyutlu Uzay Geometrisi:** $512$ boyutlu bir vektör uzayında rastgele yönlenmiş iki vektör neredeyse kesinlikle birbirine ortogonaldir (dik açılıdır). Token'ın semantik bilgisi ile pozisyon bilgisi uzayın farklı alt manifoldlarında birbirini ezmeden bir arada var olabilir.
* **Lineer Projeksiyon Ayrıştırması:** İlk lineer katman ($W_Q, W_K$), öğrenilmiş ağırlıkları sayesinde semantik bileşen ile sinüzoidal pozisyon frekanslarını kolayca birbirinden ayrıştırabilir.
