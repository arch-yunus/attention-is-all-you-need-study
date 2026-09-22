# Bölüm 11: Byte-Pair Encoding (BPE) ve Alt Kelime Tokenizasyonu

> *"Neural machine translation systems typically operate with a fixed vocabulary, but translation of rare and out-of-vocabulary words is an open problem. We address this by encoding rare words as sequences of subword units."*  
> — **Rico Sennrich et al.**, *"Neural Machine Translation of Rare Words with Subword Units" (ACL 2016)*

---

## 1. Alt Kelime (Subword) İhtiyacı

Geleneksel kelime düzeyinde tokenizasyon şu ciddi sorunlara yol açar:
1. **Devasa Kelime Haznesi (Vocab Explosion)**: Sözlükte yüzbinlerce kelime tutmak embedding katmanı boyutlarını kontrolsüzce artırır.
2. **Kelimelerin Sözlük Dışı Kalması (OOV - Out of Vocabulary)**: Eğitimde görülmeyen çekim ekleri veya yeni kelimeler `<unk>` tokenına dönüşür ve anlam kaybolur.

BPE (Byte-Pair Encoding), kelimeleri istatistiksel olarak en sık yan yana gelen karakter veya alt dizgilerine bölerek bu sorunu çözer.

---

## 2. BPE Algoritmasının Çalışma Mantığı

1. **Başlangıç**: Tüm kelimeler karakterlerine ayrılır ve sonlarına sonlandırma işareti eklenir.
   Örnek: `["d i k k a t", "d i l", "d e r i n"]`
2. **Frekans Sayımı**: Bitişik tüm karakter çiftlerinin (bigrams) sıklığı sayılır.
   Örn: `('d', 'i')` -> 2 kez, `('i', 'k')` -> 1 kez.
3. **Birleştirme (Merge)**: En yüksek frekansa sahip çift birleştirilir:
   `('d', 'i')` -> `'di'`
4. **Tekrar**: İstenen `vocab_size` hedefine ulaşılana kadar adımlar yinelenir.

---

## 3. Byte-Level BPE (GPT-2, LLaMA)

Karakter yerine doğrudan UTF-8 baytları (0..255) bazında çalışıldığında, alfabede bulunmayan herhangi bir emoji, özel sembol veya yabancı alfabe de dahil olmak üzere **asla OOV hatası oluşmaz**.
