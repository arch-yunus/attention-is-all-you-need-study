"""
BPE Tokenizer Test Paketi (tests/test_tokenizer.py).
"""

import pytest
import os
import tempfile
from src.tokenizer import BPETokenizer


def test_bpe_training_and_roundtrip():
    texts = [
        "merhaba dünya",
        "merhaba yapay zeka",
        "dikkat mekanizması",
        "transformer mimarisi",
    ]

    tokenizer = BPETokenizer()
    tokenizer.train(texts, target_vocab_size=50, min_frequency=1)

    assert tokenizer.vocab_size >= len(tokenizer.special_tokens)
    assert tokenizer.pad_token_id == 0
    assert tokenizer.bos_token_id == 1
    assert tokenizer.eos_token_id == 2
    assert tokenizer.unk_token_id == 3

    encoded = tokenizer.encode("merhaba dünya", add_special_tokens=False)
    assert len(encoded) > 0

    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    assert "merhaba" in decoded
    assert "dünya" in decoded


def test_bpe_batch_encoding_and_serialization():
    texts = ["kısa", "uzun bir metin dizisi"]
    tokenizer = BPETokenizer()
    tokenizer.train(texts, target_vocab_size=40, min_frequency=1)

    input_ids, mask = tokenizer.encode_batch(texts, max_len=10)
    assert input_ids.shape == (2, 10)
    assert mask.shape == (2, 10)

    # JSON Serileştirme
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        tokenizer.save_vocab(tmp_path)
        new_tokenizer = BPETokenizer()
        new_tokenizer.load_vocab(tmp_path)
        assert new_tokenizer.vocab_size == tokenizer.vocab_size
        assert new_tokenizer.merges == tokenizer.merges
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
