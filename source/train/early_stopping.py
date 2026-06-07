"""Early-stopping helper for the ablation study training loop.

Default ``patience=7``: training stops if the monitored metric (R@1 on
validation) has not improved by ``min_delta`` for 7 consecutive epochs.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """Track best metric, signal when to stop training.

    Parameters
    ----------
    patience : int, default 7
        Number of consecutive epochs without improvement before stopping.
    min_delta : float, default 0.0
        Minimum change in the monitored metric to qualify as improvement.
    mode : {"max", "min"}, default "max"
        Whether higher values are better (e.g. R@1) or lower (e.g. loss).

    Usage
    -----
    >>> es = EarlyStopping(patience=7)
    >>> for epoch in range(max_epochs):
    ...     train_one_epoch(...)
    ...     val_r1 = evaluate(...)
    ...     improved = es.update(val_r1)
    ...     if improved:
    ...         save_checkpoint(...)
    ...     if es.should_stop:
    ...         print(f"Early stopping at epoch {epoch}: {es.reason}")
    ...         break
    """

    patience: int = 7
    min_delta: float = 0.0
    mode: str = "max"

    def __post_init__(self):
        if self.mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        self.best: float | None = None
        self.best_epoch: int = -1
        self.epochs_without_improvement: int = 0
        self.should_stop: bool = False
        self.reason: str = ""
        self._epoch: int = -1

    # ------------------------------------------------------------------
    def _is_better(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def update(self, value: float) -> bool:
        """Record a new metric value. Returns True if it improved over the best so far."""
        self._epoch += 1
        improved = self._is_better(value)
        if improved:
            self.best = value
            self.best_epoch = self._epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                self.should_stop = True
                self.reason = (
                    f"No improvement for {self.patience} epochs "
                    f"(best={self.best:.4f} at step {self.best_epoch})"
                )
        return improved

    def state_dict(self) -> dict:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "best": self.best,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
            "should_stop": self.should_stop,
            "reason": self.reason,
            "_epoch": self._epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        for k, v in state.items():
            setattr(self, k, v)


__all__ = ["EarlyStopping"]
