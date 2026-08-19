"""Injection-side limb-darkened occultation model (used by the trial generator).

Numerically integrates the blocked stellar flux for a quadratic limb-darkening
law over a (k, d) lookup grid, then rescales so the mid-transit depth matches
the depth the trial asked for. Same logic as the old
02_generate_trials_all_stars.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import config as C


@dataclass
class LimbDarkOccultationTable:
    u1: float
    u2: float
    k_grid: np.ndarray
    d_grid: np.ndarray
    occ: np.ndarray  # (Nk, Nd): fraction of total flux blocked

    @staticmethod
    def disk_avg_intensity(u1: float, u2: float) -> float:
        return float(1.0 - u1 / 3.0 - u2 / 6.0)

    @staticmethod
    def intensity_at_r(r: np.ndarray, u1: float, u2: float) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        mu = np.sqrt(np.clip(1.0 - r * r, 0.0, 1.0))
        intensity = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
        return np.clip(intensity, 0.0, None)

    @classmethod
    def build(cls, u1: float, u2: float, k_grid: np.ndarray, d_grid: np.ndarray,
              n_r: int = 600) -> "LimbDarkOccultationTable":
        k_grid = np.asarray(k_grid, dtype=float)
        d_grid = np.asarray(d_grid, dtype=float)

        dr = 1.0 / float(n_r)
        r = (np.arange(n_r, dtype=float) + 0.5) * dr
        intensity = cls.intensity_at_r(r, u1, u2)
        F_total = np.pi * cls.disk_avg_intensity(u1, u2)

        occ = np.zeros((len(k_grid), len(d_grid)), dtype=float)
        for ik, k in enumerate(k_grid):
            k = float(k)
            if k <= 0:
                continue
            for idd, d in enumerate(d_grid):
                d = float(d)
                if d >= 1.0 + k:
                    continue
                if d <= 1e-12:
                    # Planet centred on the star: the whole disk of radius k is blocked.
                    theta = np.zeros_like(r)
                    theta[r <= k] = np.pi
                    occ[ik, idd] = float(np.sum((2.0 * theta) * r * dr * intensity) / F_total)
                    continue
                theta = np.zeros_like(r)
                if d < k:
                    theta[r <= (k - d)] = np.pi
                overlap = (r > abs(d - k)) & (r < (d + k))
                if np.any(overlap):
                    cosarg = (r[overlap] * r[overlap] + d * d - k * k) / (2.0 * r[overlap] * d)
                    theta[overlap] = np.arccos(np.clip(cosarg, -1.0, 1.0))
                occ[ik, idd] = float(np.sum((2.0 * theta) * r * dr * intensity) / F_total)
        return cls(u1=u1, u2=u2, k_grid=k_grid, d_grid=d_grid, occ=occ)

    def occult_frac(self, d: np.ndarray, k_target: float) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        k_target = float(np.clip(k_target, self.k_grid[0], self.k_grid[-1]))
        d_clip = np.clip(d, self.d_grid[0], self.d_grid[-1])

        # Locate k_target on the grid and interpolate between the two rows it falls between.
        ik = int(np.searchsorted(self.k_grid, k_target))
        if ik <= 0:
            occ_k = self.occ[0]
        elif ik >= len(self.k_grid):
            occ_k = self.occ[-1]
        else:
            k0, k1 = float(self.k_grid[ik - 1]), float(self.k_grid[ik])
            w = 0.0 if k1 == k0 else (k_target - k0) / (k1 - k0)
            occ_k = (1.0 - w) * self.occ[ik - 1] + w * self.occ[ik]
        return np.interp(d_clip, self.d_grid, occ_k, left=0.0, right=0.0)


def build_ld_table(depth_min: float, depth_max: float,
                   u1: float = C.LD_U1, u2: float = C.LD_U2) -> LimbDarkOccultationTable:
    avgI = LimbDarkOccultationTable.disk_avg_intensity(u1, u2)
    k_min = float(np.clip(np.sqrt(max(depth_min, 1e-8) * avgI) * 0.7, 0.002, 0.25))
    k_max = float(np.clip(np.sqrt(max(depth_max, 1e-8) * avgI) * 1.3, k_min * 1.05, 0.25))

    k_grid = np.linspace(k_min, k_max, C.LD_TABLE_NK)
    d_grid = np.linspace(0.0, 1.0 + k_max, C.LD_TABLE_ND)
    print(f"Building LD occultation table: u1={u1:.3f} u2={u2:.3f} "
          f"k=[{k_min:.4f},{k_max:.4f}] Nk={C.LD_TABLE_NK} Nd={C.LD_TABLE_ND} Nr={C.LD_TABLE_NR}")
    return LimbDarkOccultationTable.build(u1=u1, u2=u2, k_grid=k_grid, d_grid=d_grid,
                                          n_r=C.LD_TABLE_NR)


def transit_delta_ld(t: np.ndarray, period: float, t0: float, duration_days: float,
                     depth_mid: float, b: float,
                     ld_table: LimbDarkOccultationTable) -> np.ndarray:
    """delta(t): the fractional flux-drop series, scaled so that delta(t0) = depth_mid."""
    t = np.asarray(t, dtype=float)
    period, t0 = float(period), float(t0)
    duration_days, depth_mid, b = float(duration_days), float(depth_mid), float(b)
    if period <= 0 or duration_days <= 0:
        return np.zeros_like(t, dtype=float)

    phase = ((t - t0 + 0.5 * period) % period) - 0.5 * period

    avgI = LimbDarkOccultationTable.disk_avg_intensity(ld_table.u1, ld_table.u2)
    k_guess = float(np.clip(np.sqrt(max(depth_mid, 1e-10) * avgI),
                            ld_table.k_grid[0], ld_table.k_grid[-1]))

    x_edge = np.sqrt(max(0.0, (1.0 + k_guess) ** 2 - b * b))
    x = x_edge * (phase / (0.5 * duration_days))
    d = np.sqrt(b * b + x * x)

    in_window = np.abs(phase) <= (0.5 * duration_days)
    d_eff = np.where(in_window, d, 1.0 + k_guess + 1e-3)
    occ = ld_table.occult_frac(d_eff, k_guess)

    occ_mid = float(ld_table.occult_frac(np.array([b], dtype=float), k_guess)[0])
    scale = depth_mid / occ_mid if np.isfinite(occ_mid) and occ_mid > 0 else 0.0
    return np.clip(scale * occ, 0.0, 0.99)
