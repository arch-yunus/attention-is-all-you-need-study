"""
Eğitici ve Checkpoint Yönetimi Testleri.
"""

import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest
from src import (
    Transformer,
    Trainer,
    TrainerConfig,
    LabelSmoothingLoss,
)


@pytest.fixture
def tmp_dir():
    path = "tests_tmp_checkpoints"
    os.makedirs(path, exist_ok=True)
    yield path
    if os.path.exists(path):
        shutil.rmtree(path)


def test_trainer_fit_and_checkpoint(tmp_dir):
    torch.manual_seed(42)
    vocab_size = 20
    d_model = 32

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        d_ff=64,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = LabelSmoothingLoss(vocab_size=vocab_size, padding_idx=0, smoothing=0.1)

    config = TrainerConfig(
        epochs=3,
        device="cpu",
        save_dir=tmp_dir,
        log_interval=1,
    )
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, config=config)

    # Sentetik veri
    src = torch.randint(1, vocab_size, (16, 6))
    tgt_in = torch.randint(1, vocab_size, (16, 6))
    tgt_expected = torch.randint(1, vocab_size, (16, 6))
    dataset = TensorDataset(src, tgt_in, tgt_expected)
    loader = DataLoader(dataset, batch_size=8)

    history = trainer.fit(train_loader=loader, val_loader=loader)

    assert len(history.train_losses) == 3
    assert len(history.val_losses) == 3
    # Kayıp azalma eğiliminde olmalı
    assert history.train_losses[-1] <= history.train_losses[0]

    # Checkpoint kaydetme ve yükleme testi
    ckpt_path = trainer.save_checkpoint("test_ckpt.pt")
    assert os.path.exists(ckpt_path)

    new_model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        d_ff=64,
        dropout=0.0,
    )
    new_trainer = Trainer(model=new_model, criterion=criterion, optimizer=optimizer, config=config)
    new_trainer.load_checkpoint(ckpt_path)

    # Parametrelerin aynı olduğunu doğrula
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)
