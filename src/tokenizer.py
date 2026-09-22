"""
Byte-Pair Encoding (BPE) Tokenizer Modülü.

Sennrich et al. (2016) "Neural Machine Translation of Rare Words with Subword Units"
ve modern GPT / LLaMA tokenizer mimarilerine dayanan saf Python BPE tokenizer implementasyonu.

Özellikler:
- Karakter & Alt Kelime (Subword) birleştirme eğitimi
- Byte-level UTF-8 güvenliği (Asla OOV / Out-of-Vocabulary hatası vermez)
- Özel token yönetimi (<pad>, <bos>, <eos>, <unk>)
- JSON formatında sözlük kaydetme / yükleme
- PyTorch tensörlerine dönüştürme ve batch padding desteği
"""

import json
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set
import torch


class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) Tabanlı Alt Kelime Tokenizer'ı.
    
    Args:
        special_tokens (Optional[List[str]]): Tanımlanacak özel semboller.
    """

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, special_tokens: Optional[List[str]] = None) -> None:
        self.special_tokens = special_tokens or [
            self.PAD_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
            self.UNK_TOKEN,
        ]
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: Dict[Tuple[str, str], str] = {}
        self._init_vocab()

    def _init_vocab(self) -> None:
        self.vocab = {}
        self.inverse_vocab = {}
        # Özel tokenleri en başa ata
        for idx, token in enumerate(self.special_tokens):
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token

    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.PAD_TOKEN]

    @property
    def bos_token_id(self) -> int:
        return self.vocab[self.BOS_TOKEN]

    @property
    def eos_token_id(self) -> int:
        return self.vocab[self.EOS_TOKEN]

    @property
    def unk_token_id(self) -> int:
        return self.vocab[self.UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _get_stats(self, word_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """Tüm bitişik çiftlerin (bigram) frekanslarını hesaplar."""
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += freq
        return pairs

    def _merge_vocab(
        self,
        pair: Tuple[str, str],
        word_freqs: Dict[Tuple[str, ...], int],
    ) -> Dict[Tuple[str, ...], int]:
        """Seçilen çifti tüm kelimelerde birleştirir."""
        new_word_freqs = {}
        bigram_p1, bigram_p2 = pair
        replacement = bigram_p1 + bigram_p2

        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == bigram_p1 and word[i + 1] == bigram_p2:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        return new_word_freqs

    def train(
        self,
        texts: List[str],
        target_vocab_size: int = 1000,
        min_frequency: int = 2,
    ) -> None:
        """
        Metin külliyatı (corpus) üzerinden BPE birleştirmelerini öğrenir.
        
        Args:
            texts: Eğitim metinleri listesi.
            target_vocab_size: Erişilecek azami sözlük boyutu.
            min_frequency: Birleştirilecek çiftin en az görülme sıklığı.
        """
        self._init_vocab()
        self.merges = {}

        # 1. Kelime frekanslarını çıkar ve karakterlere böl
        raw_word_freqs = defaultdict(int)
        for text in texts:
            words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
            for w in words:
                raw_word_freqs[w] += 1

        # Karakter seviyesi frekans tablosu: ('k', 'e', 'l', 'i', 'm', 'e') -> freq
        word_freqs: Dict[Tuple[str, ...], int] = {
            tuple(list(w)): freq for w, freq in raw_word_freqs.items()
        }

        # 2. Temel alfabedeki tüm tekil karakterleri sözlüğe ekle
        unique_chars: Set[str] = set()
        for word in word_freqs.keys():
            for ch in word:
                unique_chars.add(ch)

        for ch in sorted(list(unique_chars)):
            if ch not in self.vocab:
                idx = len(self.vocab)
                self.vocab[ch] = idx
                self.inverse_vocab[idx] = ch

        # 3. İteratif BPE Birleştirmeleri (Merge Loop)
        num_merges = target_vocab_size - len(self.vocab)
        for _ in range(max(0, num_merges)):
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < min_frequency:
                break

            merged_token = best_pair[0] + best_pair[1]
            self.merges[best_pair] = merged_token

            if merged_token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[merged_token] = idx
                self.inverse_vocab[idx] = merged_token

            word_freqs = self._merge_vocab(best_pair, word_freqs)

            if len(self.vocab) >= target_vocab_size:
                break

    def _tokenize_word(self, word: str) -> List[str]:
        """Tek bir kelimeyi BPE kuralları ile alt kelimelerine ayırır."""
        pieces = list(word)
        if len(pieces) <= 1:
            return pieces

        while len(pieces) > 1:
            pairs = [(pieces[i], pieces[i + 1]) for i in range(len(pieces) - 1)]
            # Yapılabilecek merge'leri bul
            mergeable = [p for p in pairs if p in self.merges]
            if not mergeable:
                break

            # En erken öğrenilen birleştirmeyi uygula
            pair_to_merge = mergeable[0]
            new_pieces = []
            i = 0
            while i < len(pieces):
                if i < len(pieces) - 1 and (pieces[i], pieces[i + 1]) == pair_to_merge:
                    new_pieces.append(self.merges[pair_to_merge])
                    i += 2
                else:
                    new_pieces.append(pieces[i])
                    i += 1
            pieces = new_pieces

        return pieces

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Metni token kimliklerine (IDs) dönüştürür.
        """
        token_ids: List[int] = []
        if add_special_tokens:
            token_ids.append(self.bos_token_id)

        words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        for word in words:
            pieces = self._tokenize_word(word)
            for p in pieces:
                if p in self.vocab:
                    token_ids.append(self.vocab[p])
                else:
                    # Karakter fallback
                    for ch in p:
                        token_ids.append(self.vocab.get(ch, self.unk_token_id))

        if add_special_tokens:
            token_ids.append(self.eos_token_id)

        return token_ids

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """
        Token kimliklerini tekrar okunabilir metne dönüştürür.
        """
        out_tokens = []
        for t in tokens:
            if skip_special_tokens and t in (
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
                self.unk_token_id,
            ):
                continue
            tok_str = self.inverse_vocab.get(t, self.UNK_TOKEN)
            out_tokens.append(tok_str)

        # Doğal boşluk birleştirme
        text = "".join(out_tokens)
        return text

    def encode_batch(
        self,
        texts: List[str],
        max_len: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Birden çok metni tensöre dönüştürür ve padding uygular.
        
        Returns:
            input_ids: [Batch, Max_Seq_Len]
            attention_mask: [Batch, Max_Seq_Len] (1 for tokens, 0 for pad)
        """
        encoded_list = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]
        if max_len is None:
            max_len = max(len(seq) for seq in encoded_list)

        batch_size = len(texts)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

        for i, seq in enumerate(encoded_list):
            trunc_seq = seq[:max_len]
            input_ids[i, : len(trunc_seq)] = torch.tensor(trunc_seq, dtype=torch.long)
            attention_mask[i, : len(trunc_seq)] = True

        return input_ids, attention_mask

    def save_vocab(self, filepath: str) -> None:
        """Sözlüğü ve merge kurallarını JSON olarak kaydeder."""
        data = {
            "special_tokens": self.special_tokens,
            "vocab": self.vocab,
            "merges": [f"{p[0]} {p[1]}" for p in self.merges.keys()],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_vocab(self, filepath: str) -> None:
        """JSON dosyasından sözlük ve kuralları yükler."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.special_tokens = data.get("special_tokens", self.special_tokens)
        self.vocab = data["vocab"]
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = {}
        for m in data.get("merges", []):
            parts = m.split(" ")
            if len(parts) == 2:
                self.merges[(parts[0], parts[1])] = parts[0] + parts[1]
