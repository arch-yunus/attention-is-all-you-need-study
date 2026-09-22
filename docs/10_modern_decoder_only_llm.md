# Bölüm 10: Modern Salt-Dekoder (Decoder-Only) LLM Mimarisi

> *"Generative Pre-trained Transformer demonstrates that task-agnostic language models can be used across a wide variety of tasks with zero-shot generalization."*  
> — **Radford et al.**, *"Language Models are Unsupervised Multitask Learners (GPT-2, OpenAI 2019)"*

---

## 1. Encoder-Decoder'dan Decoder-Only Mimarisine Evrim

Vaswani vd. (2017) makalesi İngilizce-Almanca makine çevirisi için tasarlandığından iki parçalı (Encoder-Decoder) bir yapıya sahipti:
- **Encoder**: Girdi dizisini çift yönlü (bidirectional) olarak anlar.
- **Cross-Attention**: Decoder'ın Encoder temsillerine erişmesini sağlar.
- **Decoder**: Otoregresif olarak hedef dili üretir.

Ancak GPT serisi (Radford et al.) ve günümüzün tüm öncü açık kaynaklı modelleri (LLaMA-3, Mistral, Gemma, Falcon, Qwen) **Decoder-Only** (Salt-Dekoder) mimarisine geçmiştir:

```
[2017 Seq2Seq Encoder-Decoder]           [2024-2026 Decoder-Only Modern LLM]
  Input -> [Encoder] -> Memory                Prompt + Context
                            |                       |
                            v                       v
  Output -> [Masked Dec] -> [Cross-Attn] -> [RoPE + GQA + RMSNorm + SwiGLU]
                            |                       |
                            v                       v
                         Logits               Next-Token Logits (Autoregressive)
```

---

## 2. Modern Decoder-Only Bloğunun Anatomisi

Modern bir LLM katmanı (örn. LLaMA-3 / Mistral), aşağıdaki 5 temel iyileştirmeyi birleştirir:

1. **Pre-LN Mimarisi**: Normalizasyon alt katmanların önüne taşınır, böylece artık bağlantılar doğrudan gradyan otoyolu oluşturur.
2. **RMSNorm**: Klasik LayerNorm yerine ortalamasız kök kare normalizasyonu.
3. **Rotary Positional Embedding (RoPE)**: Mutlak sinüzoidal dalgalar yerine 2B düzlemlerde açısal koordinat döndürme.
4. **Grouped-Query Attention (GQA)**: KV-Cache boyutunu 4x-8x azaltarak çıkarım maliyetini düşürme.
5. **SwiGLU FFN**: ReLU yerine kapılı SiLU aktivasyonu.

---

## 3. Matematiksel Akış

Bir $l$-inci katman için girdi $x_l \in \mathbb{R}^{B \times S \times d_{\text{model}}}$ olmak üzere:

$$h_l = \text{RMSNorm}(x_l)$$
$$q, k, v = h_l W_q, \; h_l W_k, \; h_l W_v$$
$$q_{\text{rot}}, k_{\text{rot}} = \text{RoPE}(q), \; \text{RoPE}(k)$$
$$\text{attn\_out} = \text{GQA}(q_{\text{rot}}, k_{\text{rot}}, v, \text{CausalMask}) W_o$$
$$x_{l, \text{mid}} = x_l + \text{attn\_out}$$
$$\text{ffn\_out} = \text{SwiGLU}(\text{RMSNorm}(x_{l, \text{mid}}))$$
$$x_{l+1} = x_{l, \text{mid}} + \text{ffn\_out}$$

---

## 4. Kayıp Fonksiyonu ve Hedef Öteleme (Shifted Cross-Entropy)

Otoregresif dil modellemesinde girdi token dizisi $T = (t_1, t_2, \dots, t_N)$ için model $t_{i+1}$'i tahmin etmeye çalışır:

$$\mathcal{L} = -\frac{1}{N-1} \sum_{i=1}^{N-1} \log P(t_{i+1} \mid t_1, t_2, \dots, t_i)$$

Kod düzeyinde girdiler $T_{0:N-1}$, hedefler ise $T_{1:N}$ şeklinde bir adım ötelenerek Cross-Entropy loss hesaplanır.
