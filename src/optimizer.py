"""
Noam Öğrenme Oranı Zamanlayıcısı (Noam Learning Rate Scheduler).

"Attention Is All You Need" (Vaswani et al., 2017) Bölüm 5.3:
lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler


class NoamLR(_LRScheduler):
    """
    Orijinal Transformer Noam Öğrenme Oranı Çizelgeleyicisi.

    İlk `warmup_steps` adım boyunca öğrenme oranını doğrusal olarak artırır;
    sonrasında ise adım sayısının ters karekökü (1 / sqrt(step)) ile orantılı
    olarak kademeli şekilde düşürür.

    Args:
        optimizer (torch.optim.Optimizer): Optimize edici nesnesi (genellikle Adam).
        d_model (int): Modelin gizli boyutu. Varsayılan: 512.
        warmup_steps (int): Isınma (warmup) adım sayısı. Varsayılan: 4000.
        factor (float): Genel ölçekleme katsayısı. Varsayılan: 1.0.
        last_epoch (int): Son epoch/adım indeksi. Varsayılan: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int = 512,
        warmup_steps: int = 4000,
        factor: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch)
        scale = self.factor * (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        return [scale for _ in self.base_lrs]
