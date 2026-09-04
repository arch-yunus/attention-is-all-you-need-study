"""
Eğitim ve Değerlendirme Yöneticisi (Trainer & Checkpoint Engine).

Uçtan uca model eğitimini, doğrulama adımlarını, gradyan kırpmayı (gradient clipping),
öğrenme oranı zamanlamasını ve model kontrol noktası (checkpoint) kaydetmeyi yönetir.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .transformer import Transformer
from .masks import create_masks
from .metrics import calculate_perplexity


@dataclass
class TrainerConfig:
    """Eğitici yapılandırma parametreleri."""
    epochs: int = 10
    learning_rate: float = 1e-3
    grad_clip_norm: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pad_idx: int = 0
    save_dir: str = "checkpoints"
    save_best_only: bool = True
    log_interval: int = 10


@dataclass
class TrainingHistory:
    """Eğitim metrik geçmişi."""
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    perplexities: List[float] = field(default_factory=list)


class Trainer:
    """
    Modüler Transformer Eğitici Sınıfı.
    """

    def __init__(
        self,
        model: Transformer,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[TrainerConfig] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        self.config = config or TrainerConfig()
        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.history = TrainingHistory()
        self.best_val_loss = float("inf")

        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir, exist_ok=True)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Tek bir epoch boyunca modeli eğitir."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        for step, batch in enumerate(dataloader):
            if len(batch) == 3:
                src, tgt_in, tgt_expected = batch
            elif len(batch) == 2:
                src, tgt = batch
                tgt_in = tgt[:, :-1]
                tgt_expected = tgt[:, 1:]
            else:
                raise ValueError("Batch (src, tgt_in, tgt_expected) veya (src, tgt) formatında olmalıdır!")

            src = src.to(self.device)
            tgt_in = tgt_in.to(self.device)
            tgt_expected = tgt_expected.to(self.device)

            src_mask, tgt_mask = create_masks(src, tgt_in, pad_idx=self.config.pad_idx)

            self.optimizer.zero_grad()
            logits = self.model(src, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask)

            # Kayıp hesabı
            loss = self.criterion(logits, tgt_expected)
            loss.backward()

            # Gradyan Kırpma (Gradient Clipping - Vaswani et al. 2017)
            if self.config.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Modeli doğrulama veri kümesinde değerlendirir."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)

        for batch in dataloader:
            if len(batch) == 3:
                src, tgt_in, tgt_expected = batch
            else:
                src, tgt = batch
                tgt_in = tgt[:, :-1]
                tgt_expected = tgt[:, 1:]

            src = src.to(self.device)
            tgt_in = tgt_in.to(self.device)
            tgt_expected = tgt_expected.to(self.device)

            src_mask, tgt_mask = create_masks(src, tgt_in, pad_idx=self.config.pad_idx)
            logits = self.model(src, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask)
            loss = self.criterion(logits, tgt_expected)
            total_loss += loss.item()

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callable[[int, float, Optional[float]], None]]] = None,
    ) -> TrainingHistory:
        """
        Modeli yapılandırılmış epoch sayısı boyunca eğitir ve doğrular.
        """
        for epoch in range(1, self.config.epochs + 1):
            start_time = time.time()
            train_loss = self.train_epoch(train_loader)
            self.history.train_losses.append(train_loss)

            val_loss = None
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                self.history.val_losses.append(val_loss)
                ppl = calculate_perplexity(val_loss)
                self.history.perplexities.append(ppl)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(f"best_model_loss_{val_loss:.4f}.pt")

            # LR geçmişi
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history.learning_rates.append(current_lr)

            elapsed = time.time() - start_time

            if epoch % self.config.log_interval == 0 or epoch == self.config.epochs:
                val_str = f" | Val Loss: {val_loss:.4f} | PPL: {calculate_perplexity(val_loss):.2f}" if val_loss else ""
                print(f"Epoch [{epoch:02d}/{self.config.epochs:02d}] ({elapsed:.1f}s) | Train Loss: {train_loss:.4f}{val_str} | LR: {current_lr:.6f}")

            if callbacks:
                for cb in callbacks:
                    cb(epoch, train_loss, val_loss)

        return self.history

    def save_checkpoint(self, filename: str) -> str:
        """Model ve eğitici durumunu diske kaydeder."""
        filepath = os.path.join(self.config.save_dir, filename)
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history.__dict__,
            "best_val_loss": self.best_val_loss,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, filepath)
        return filepath

    def load_checkpoint(self, filepath: str) -> None:
        """Kaydedilmiş kontrol noktasını yükler."""
        state = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        if "best_val_loss" in state:
            self.best_val_loss = state["best_val_loss"]
