"""
Noam Optimizer, Label Smoothing ve Pre-LN Katman Testleri.
"""

import pytest
import torch
import torch.nn as nn
from src import (
    NoamLR,
    LabelSmoothingLoss,
    LayerNorm,
    SublayerConnection,
    Transformer,
)


def test_noam_lr_schedule():
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    warmup_steps = 400
    scheduler = NoamLR(optimizer, d_model=512, warmup_steps=warmup_steps)

    lrs = []
    for step in range(1, 1000):
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    # Warmup süresince LR artmalıdır
    assert lrs[warmup_steps - 2] > lrs[0], "Warmup sırasında LR artmalı!"

    # Warmup zirvesinden sonra LR kademeli azalmalıdır (1 / sqrt(step))
    assert lrs[-1] < lrs[warmup_steps], "Warmup sonrasında LR azalmalı!"


def test_label_smoothing_loss():
    vocab_size = 10
    pad_idx = 0
    criterion = LabelSmoothingLoss(vocab_size=vocab_size, padding_idx=pad_idx, smoothing=0.1)

    logits = torch.randn(4, 5, vocab_size)
    targets = torch.randint(1, vocab_size, (4, 5))
    targets[0, -1] = pad_idx  # Bir token pad olsun

    loss = criterion(logits, targets)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)


def test_pre_ln_vs_post_ln():
    src_vocab = 30
    tgt_vocab = 30
    d_model = 32

    # Hem post_ln hem pre_ln başarıyla derlenip çalışmalı
    model_post = Transformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=d_model,
        norm_type="post_ln",
    )
    model_pre = Transformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=d_model,
        norm_type="pre_ln",
    )

    src = torch.randint(1, src_vocab, (2, 4))
    tgt = torch.randint(1, tgt_vocab, (2, 4))

    out_post = model_post(src, tgt, pad_idx=0)
    out_pre = model_pre(src, tgt, pad_idx=0)

    assert out_post.shape == (2, 4, tgt_vocab)
    assert out_pre.shape == (2, 4, tgt_vocab)
