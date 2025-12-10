from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..utils.math_utils import ema_update


@dataclass
class RecorderConfig:
    # LOW band is "pathologically depolarized" (more positive). We assume low_exit < low_enter.
    low_enter: float = -12.0  # mV threshold to ENTER LOW: V >= low_enter
    low_exit: float = -18.0   # mV threshold to EXIT  LOW: V <= low_exit
    tau_low: float = 600.0    # seconds, leaky integrator time constant for LOW occupancy
    mismatch_threshold: float = 5.0  # arbitrary units; usage defined by caller


class Recorder:
    """
    Recorder maintains LOW-band state with Schmitt hysteresis, a leaky LOW occupancy
    integrator, and neighbor mismatch. It also exposes domain_low_fraction and a
    global_v_offset (EMA of mean(V) relative to a healthy reference).
    """

    def __init__(self, cfg: RecorderConfig, grid: Tuple[int, int], dt: float):
        self.cfg = cfg
        self.h, self.w = grid
        self.dt = float(dt)

        # Schmitt LOW state (boolean grid)
        self.low_state = np.zeros((self.h, self.w), dtype=bool)

        # Leaky LOW occupancy state (float grid)
        self.low_occ = np.zeros((self.h, self.w), dtype=float)
        self._low_alpha = float(
            np.clip(self.dt / max(1e-6, self.cfg.tau_low), 1e-6, 1.0)
        )

        # EMA of global voltage offset
        self._ema_global_offset = 0.0
        self._healthy_ref: float | None = None
        # ~10 min EMA default if dt is seconds
        self._ema_alpha = float(np.clip(self.dt / 600.0, 1e-6, 1.0))

    def update_bands(self, V: np.ndarray) -> None:
        """
        Update LOW-band Schmitt state given current V (mV).

        LOW = pathologically depolarized band. We assume low_exit < low_enter.

        - Enter LOW when V >= low_enter (more depolarized).
        - Exit  LOW when V <= low_exit (more hyperpolarized).
        - Otherwise keep previous state.
        """
        assert V.shape == (self.h, self.w)
        enter = V >= self.cfg.low_enter
        exit_ = V <= self.cfg.low_exit
        self.low_state = np.where(
            enter,
            True,
            np.where(exit_, False, self.low_state),
        )

    def update_low_occupancy(self) -> None:
        """Leaky integrate LOW occupancy indicator into low_occ (unitless)."""
        indicator = self.low_state.astype(float)
        self.low_occ = (1.0 - self._low_alpha) * self.low_occ + self._low_alpha * indicator

    def neighbor_mismatch(self, V: np.ndarray) -> np.ndarray:
        """
        Compute a simple neighbor mismatch metric on LOW-state labels.
        For each cell: fraction of 4-neighbors that differ from the cell's LOW label.
        """
        ls = self.low_state.astype(float)
        up = np.roll(ls, -1, axis=0)
        down = np.roll(ls, 1, axis=0)
        left = np.roll(ls, 1, axis=1)
        right = np.roll(ls, -1, axis=1)
        neigh_mean = (up + down + left + right) / 4.0
        # If cell is LOW, mismatch = 1 - neigh_mean; else mismatch = neigh_mean
        mismatch = np.where(self.low_state, 1.0 - neigh_mean, neigh_mean)
        return mismatch

    def domain_low_fraction(self) -> float:
        """Fraction of cells currently in LOW band."""
        return float(np.mean(self.low_state))

    def set_healthy_ref(self, v_ref: float) -> None:
        self._healthy_ref = float(v_ref)

    def global_v_offset(self, V: np.ndarray) -> float:
        """
        EMA of mean(V) - healthy_ref, using a slow time-constant.
        If healthy_ref not set, default to the current mean(V) on first call.
        """
        if self._healthy_ref is None:
            self._healthy_ref = float(np.mean(V))
        delta = float(np.mean(V) - self._healthy_ref)
        self._ema_global_offset = ema_update(self._ema_global_offset, delta, self._ema_alpha)
        return self._ema_global_offset


