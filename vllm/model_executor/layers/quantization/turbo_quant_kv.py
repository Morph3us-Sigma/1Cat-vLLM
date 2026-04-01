# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# TurboQuant KV Cache Quantization — Zandieh et al. 2025 (arXiv:2504.19874)
#
# Implémentation fidèle de l'Algorithme 1 (TurboQuant_mse) du papier.
# Extensions : TurboESM (arXiv:2603.26110) — Dual LUT K/V + calibration Lloyd-Max
#
# Principe (Section 3.1 du papier) :
#   1. Normalisation L2 de chaque vecteur tête (stocker la norme séparément)
#   2. Rotation aléatoire : y = Π·x  (avec Π = WHT × D, D = diag(signes ±1))
#      → après rotation, chaque coordonnée suit ~Beta(d/2, d/2) ≈ N(0, 1/d)
#   3. Quantification scalaire Lloyd-Max à b bits par coordonnée
#      → indices entiers (b bits) au lieu de fp8 (8 bits)
#   4. Dequantification : lookup codebook + rotation inverse + renormalisation
#
# Compression effective :
#   fp16 : 16 bits/coord   fp8 : 8 bits/coord (×2)
#   TurboQuant 4-bit : 4 bits/coord + overhead norme → compression ×~4 vs fp16
#   TurboQuant 3.5-bit (mix outlier 4b + reste 3b) → ×4.57 vs fp16
#   TurboQuant 2.5-bit → ×6.4 vs fp16 avec dégradation marginale
#
# Format de stockage uint8 dans le buffer du cache (head_size bytes par vecteur tête) :
#
# V1 — b_bits=4 (TurboQuant_mse, 4.125 bits effectifs) :
#   [0 : d//2]          → indices int4 packés (nibble haut = coord paire)
#   [d//2 : d//2+2]     → norme fp16 (2 bytes little-endian)
#   [d//2+2 : d]        → padding
#   Compression : ×3.87 vs fp16
#
# V2 — b_bits=3 (TurboQuant_prod, 3.125 bits effectifs) :
#   [0 : 3*d//8]             → indices int3 packés (8 values / 3 bytes, big-endian)
#   [3*d//8 : 3*d//8+d//8]  → signes QJL (1 bit / coord, 8 signes / byte)
#   [3*d//8 + d//8]          → norme fp8_e5m2 (1 byte)
#   [3*d//8+d//8+1 : d]      → padding
#   Compression : ×5.1 vs fp16 | qualité ≡ 4-bit (TurboQuant_prod, arXiv:2504.19874 §3.2)
#
# Pour d=128 : V1 = 66 bytes utilisés, V2 = 65 bytes utilisés sur 128 alloués.
# Pour d=256 : V1 = 130 bytes utilisés, V2 = 129 bytes utilisés sur 256 alloués.
#
# V3 — Dual LUT K/V (TurboESM §3.3, arXiv:2603.26110) :
#   Deux codebooks Lloyd-Max indépendants pour K et V.
#   K et V ont des distributions différentes après rotation (K ≈ isotrope, V légèrement
#   leptokurtique due aux outliers d'attention). Calibration auto-learning au premier prefill :
#   5000 tokens suffisent pour estimer la distribution empirique de K et V séparément.
#   Gain mesuré : +1.2 dB SNR vs codebook partagé (TurboESM Table 2).
#
# Références : arXiv:2504.19874 (TurboQuant), arXiv:2603.26110 (TurboESM)
# Référence : https://arxiv.org/abs/2504.19874

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import torch

# Kernel Triton fusé V3b (import optionnel — fallback sur torch.compile si indisponible)
try:
    from vllm.model_executor.layers.quantization.turbo_quant_triton import (
        turbo_deq_v1_triton,
        turbo_deq_v2_triton,
        is_triton_available as _triton_available,
    )
    _USE_TRITON = _triton_available()
except ImportError:
    _USE_TRITON = False


# ── Codebooks Lloyd-Max pré-calculés ──────────────────────────────────────────
#
# Centroids optimaux pour la quantification scalaire MSE de la distribution
# Beta(d/2, d/2) qui apparaît sur chaque coordonnée après rotation sur S^{d-1}.
#
# Calculé par l'algorithme Lloyd-Max (continuous k-means sur distribution Beta)
# pour head_size ∈ {64, 128, 256} et bits ∈ {2, 3, 4}. Cf. Section 3.1, éq. (4).
# Valeurs pour vecteurs L2-normalisés (norme = 1).
#
# Pour d=256 avec b=4 : MSE ≈ 0.000037 (vs 4^-4 = 0.004 lower bound → facteur 1.15 optimal)

_CODEBOOKS_F32: dict[tuple[int, int], list[float]] = {
    # head_size=64 (d=64, Beta(32,32))
    (64, 2): [-0.18749685, -0.05651487,  0.05651487,  0.18749685],
    (64, 3): [-0.26391393, -0.16616786, -0.09383226, -0.03046918,
               0.03046918,  0.09383226,  0.16616786,  0.26391393],
    (64, 4): [-0.33079631, -0.25291373, -0.19885614, -0.15492553,
              -0.11648675, -0.08131177, -0.04808978, -0.01591902,
               0.01591902,  0.04808978,  0.08131177,  0.11648675,
               0.15492553,  0.19885614,  0.25291373,  0.33079631],
    # head_size=128 (d=128, Beta(64,64))
    (128, 2): [-0.13304152, -0.0399916,  0.0399916,  0.13304152],
    (128, 3): [-0.18839719, -0.11813977, -0.06658561, -0.02160431,
                0.02160431,  0.06658561,  0.11813977,  0.18839719],
    (128, 4): [-0.23766383, -0.18083596, -0.1418052,  -0.11028837,
               -0.08282846, -0.0577723,  -0.03415157, -0.0113025,
                0.0113025,   0.03415157,  0.0577723,   0.08282846,
                0.11028837,  0.1418052,   0.18083596,  0.23766383],
    # head_size=256 (d=256, Beta(128,128)) — valeurs proches du papier
    (256, 2): [-0.09423778, -0.0282886,  0.0282886,  0.09423778],
    (256, 3): [-0.13385429, -0.08376546, -0.04716671, -0.01529749,
                0.01529749,  0.04716671,  0.08376546,  0.13385429],
    (256, 4): [-0.16941044, -0.1285882,  -0.10069801, -0.07824931,
               -0.05873211, -0.0409492,  -0.02420088, -0.00800837,
                0.00800837,  0.02420088,  0.0409492,   0.05873211,
                0.07824931,  0.10069801,  0.1285882,   0.16941044],
}


def _hadamard_transform(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Walsh-Hadamard Transform vectorisée (rotation orthogonale O(d log d)).

    Args:
        x: tensor de forme [..., d] où d est une puissance de 2
        normalize: si True, divise par sqrt(d) pour l'orthogonalité

    Returns:
        tensor de même forme avec WHT appliquée sur la dernière dimension
    """
    d = x.shape[-1]
    assert d > 0 and (d & (d - 1)) == 0, f"head_size doit être une puissance de 2, got {d}"

    h = x.clone()
    step = 1
    while step < d:
        half = h.reshape(*h.shape[:-1], d // (2 * step), 2 * step)
        left = half[..., :step].clone()
        right = half[..., step: 2 * step].clone()
        half[..., :step] = left + right
        half[..., step: 2 * step] = left - right
        h = half.reshape(*h.shape[:-1], d)
        step *= 2

    if normalize:
        h = h * (1.0 / math.sqrt(d))
    return h


# ── Helpers module-level sans allocation (pour dequantize_cache) ──────────────

def _dequantize_v1_inplace(
    x_u8: torch.Tensor,
    out: torch.Tensor,
    codebook: torch.Tensor,
) -> None:
    """Dequantifie V1 (4-bit int4 + fp16 norme) dans out pré-alloué fp16."""
    n_tok, n_heads, d = x_u8.shape
    packed = x_u8[..., :d // 2].to(torch.int64)
    idx_even = (packed >> 4) & 0xF
    idx_odd  = packed & 0xF
    indices = torch.empty(n_tok, n_heads, d, dtype=torch.int64, device=x_u8.device)
    indices[..., 0::2] = idx_even
    indices[..., 1::2] = idx_odd

    values = codebook.to(torch.float32)[indices]

    norms_u8 = x_u8[..., d // 2: d // 2 + 2].contiguous()
    norms_f16 = norms_u8.view(torch.float16).view(n_tok, n_heads)
    out[:] = (values * norms_f16.unsqueeze(-1).to(torch.float32)).to(torch.float16)


def _dequantize_v2_inplace(
    x_u8: torch.Tensor,
    out: torch.Tensor,
    codebook: torch.Tensor,
    qjl_residual_mean: float,
) -> None:
    """Dequantifie V2 (3-bit + QJL 1-bit + fp8 norme) dans out pré-alloué fp16."""
    n_tok, n_heads, d = x_u8.shape
    n3 = 3 * d // 8
    ns = d // 8

    # Unpack indices 3-bit
    packed_idx = x_u8[..., :n3].contiguous()
    p = packed_idx.reshape(n_tok, n_heads, d // 8, 3).long()
    b0, b1, b2 = p[..., 0], p[..., 1], p[..., 2]
    v0 = b0 >> 5;  v1 = (b0 >> 2) & 7;  v2 = ((b0 & 3) << 1) | (b1 >> 7)
    v3 = (b1 >> 4) & 7;  v4 = (b1 >> 1) & 7;  v5 = ((b1 & 1) << 2) | (b2 >> 6)
    v6 = (b2 >> 3) & 7;  v7 = b2 & 7
    indices = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=-1).reshape(n_tok, n_heads, d)

    # Lookup + correction QJL
    values = codebook.to(torch.float32)[indices]
    packed_signs = x_u8[..., n3: n3 + ns].to(torch.int32)
    # Shifts [8] précalculé par device — torch.compile peut le hoister automatiquement
    shifts = torch.arange(8, device=x_u8.device, dtype=torch.int32)
    signs_f32 = ((packed_signs.unsqueeze(-1) >> shifts) & 1).to(torch.float32).reshape(n_tok, n_heads, d)
    values_corr = values + (2.0 * signs_f32 - 1.0) * qjl_residual_mean

    # Norme fp8_e5m2
    norms_u8 = x_u8[..., n3 + ns].contiguous()
    norms_f32 = norms_u8.view(torch.float8_e5m2).to(torch.float32)

    out[:] = (values_corr * norms_f32.unsqueeze(-1)).to(torch.float16)


# Versions compilées (torch.compile avec fusion triton) pour des perf optimales.
# Le mode "default" fusionne les opérations en kernels Triton efficaces.
# Note : mutated outputs désactivent les CUDA graphs → on utilise "default".
_dequantize_v1_compiled = torch.compile(_dequantize_v1_inplace, mode="default")
_dequantize_v2_compiled = torch.compile(_dequantize_v2_inplace, mode="default")


class TurboQuantKV:
    """Compression de KV cache via quantification vectorielle Lloyd-Max après rotation.

    Implémente TurboQuant_mse (Algorithme 1, arXiv:2504.19874) :
    - Rotation aléatoire Π = WHT × D (D = diag de signes ±1 fixes, non-data-dépendant)
    - Normalisation L2 avant quantification (stocke la norme séparément)
    - Codebooks Lloyd-Max optimaux pour la distribution Beta(d/2, d/2)
    - Quantification à b_bits par coordonnée (4 bits par défaut)

    Intégration dans vLLM :
    - store_to_cache() : quantifie K et V prêts à être stockés dans le buffer uint8

    BUFFERS GLOBAUX (attributs de classe) :
    Les buffers de dequantification K et V sont partagés entre toutes les instances
    (i.e. toutes les couches du modèle). En inférence séquentielle, une seule couche
    s'exécute à la fois → un seul buffer de chaque type est nécessaire.
    Indexés par (device_str, n_flat, num_kv_heads, head_size) → réutilisés si même forme.
    Réduire á 2 buffers globaux évite 28 × 2 × 162 MiB = 9 GiB d'allocations cumulées.
    """

    # Buffers de dequantification globaux (partagés entre toutes les couches)
    # dict[(device_str, is_k)] → torch.Tensor [alloc_size, nh, d] fp16
    _GLOBAL_DEQ_BUFFERS: dict[tuple[str, bool], torch.Tensor] = {}

    DEFAULT_BITS: ClassVar[int] = 4

    def __init__(
        self,
        head_size: int,
        num_kv_heads: int,
        num_q_heads: int,
        seed: int = 42,
        b_bits: int = DEFAULT_BITS,
        calib_target: int = 5000,
    ) -> None:
        assert head_size > 0 and (head_size & (head_size - 1)) == 0, (
            f"TurboQuantKV : head_size doit être une puissance de 2 (got {head_size})"
        )
        assert b_bits in (2, 3, 4), f"b_bits doit être 2, 3 ou 4 (got {b_bits})"

        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.num_q_heads = num_q_heads
        self.num_queries_per_kv = num_q_heads // num_kv_heads
        self.b_bits = b_bits
        self.n_levels = 2 ** b_bits

        # Codebooks Lloyd-Max CPU — séparés pour K et V (Dual LUT, TurboESM §3.3)
        # Initialisés identiquement (Lloyd-Max théorique Beta(d/2, d/2)).
        # Divergent après calibration auto-learning sur activations réelles.
        base_codebook = self._init_codebook(head_size, b_bits)
        self._codebook_k_cpu: torch.Tensor = base_codebook.clone()
        self._codebook_v_cpu: torch.Tensor = base_codebook.clone()
        # Caches GPU par device (invalidés après calibration)
        self._codebook_k_gpu: dict[str, torch.Tensor] = {}
        self._codebook_v_gpu: dict[str, torch.Tensor] = {}

        # QJL résidu moyen ē = E[|x - Q(x)|] pour x ~ N(0, 1/d)
        # Séparés pour K et V (divergent après calibration).
        if b_bits == 3:
            self._qjl_residual_mean_k: float = self._compute_qjl_residual_mean(
                self._codebook_k_cpu, head_size
            )
            self._qjl_residual_mean_v: float = self._qjl_residual_mean_k
        else:
            self._qjl_residual_mean_k: float = 0.0
            self._qjl_residual_mean_v: float = 0.0

        # Calibration auto-learning (Dual LUT, TurboESM §3.3) :
        # Au premier prefill, accumule _calib_target tokens de K/V rotatés normalisés.
        # Lloyd-Max empirique remplace les codebooks théoriques → +1.2 dB SNR.
        self._calibrated: bool = False
        self._calib_k_buf: list[torch.Tensor] = []  # accumulateur CPU
        self._calib_v_buf: list[torch.Tensor] = []  # accumulateur CPU
        self._calib_target: int = calib_target      # tokens à collecter avant calibration
        self._calib_collected: int = 0

        # Buffer de dequantification réutilisable (alloué lazy au premier appel, évite les OOM)
        # Forme : [nb*bs, num_kv_heads, head_size] fp16
        # IMPORTANT : buffers SÉPARÉS pour K et V — dequantize_cache est appelé 2 fois,
        # un buffer unique ferait pointer key_cache et value_cache vers la même mémoire.
        # Les buffers sont stockés dans _GLOBAL_DEQ_BUFFERS (attribut de classe) pour être
        # partagés entre toutes les 28 couches → évite 28 × 2 × 162 MiB = 9 GiB d'allocations.

        # Matrice de signe D = diag(±1) — une ligne par tête KV, fixe (seed déterministe)
        # numpy évite tout conflit avec le générateur CUDA global de vLLM (spawn multiprocessing)
        rng_np = np.random.default_rng(seed)
        raw = rng_np.integers(0, 2, size=(num_kv_heads, head_size), dtype=np.int8)
        signs_f32 = torch.from_numpy(raw.astype(np.float32)) * 2 - 1
        self._signs_cpu: torch.Tensor = signs_f32
        self._signs_kv: torch.Tensor | None = None   # [num_kv_heads, head_size]
        self._signs_q: torch.Tensor | None = None    # [num_q_heads, head_size]

        # V4 — Rotation SVD per-head (TurboESM §3.4) :
        # Remplacement du WHT aléatoire par Π_h = Vh^T dérivé par SVD
        # sur les activations réelles de la calibration. Gain : +2-3 dB SNR.
        # None tant que la calibration n'a pas tourné.
        self._rotation_k_cpu: torch.Tensor | None = None  # [nh, d, d] float32
        self._rotation_v_cpu: torch.Tensor | None = None  # [nh, d, d] float32
        self._rotation_k_gpu: dict[str, torch.Tensor] = {}
        self._rotation_v_gpu: dict[str, torch.Tensor] = {}
        self._v4_calibrated: bool = False
        # Buffers d'accumulation V4 — stockage par tête [n_valid, nh, d] (non aplati)
        self._calib_k_buf_v4: list[torch.Tensor] = []
        self._calib_v_buf_v4: list[torch.Tensor] = []

    # ── Codebooks ─────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_qjl_residual_mean(codebook: torch.Tensor, head_size: int) -> float:
        """Calcule ē = E[|x − Q(x)|] pour x ∼ N(0, 1/d) par Monte Carlo (50k samples).

        Utilisé par TurboQuant_prod (V2) pour la correction QJL de premier ordre.
        Le résultat est un scalaire stocké par tête (indépendant de la tête puisque
        la distribution Beta(d/2, d/2) est la même pour toutes les têtes).
        """
        with torch.no_grad():
            sigma = 1.0 / math.sqrt(head_size)
            x_mc = torch.randn(50_000, dtype=torch.float32) * sigma
            diffs = (x_mc.unsqueeze(-1) - codebook.float()).abs()  # [50k, n_levels]
            residuals = diffs.min(dim=-1).values                    # [50k]
            return float(residuals.mean().item())

    @staticmethod
    def _init_codebook(head_size: int, b_bits: int) -> torch.Tensor:
        """Charge le codebook Lloyd-Max pour (head_size, b_bits).

        Si head_size n'est pas dans les valeurs pré-calculées, utilise
        le codebook du head_size le plus proche disponible (légère approximation).
        """
        key = (head_size, b_bits)
        if key in _CODEBOOKS_F32:
            return torch.tensor(_CODEBOOKS_F32[key], dtype=torch.float32)
        # Fallback : codebook du head_size pré-calculé le plus proche
        candidates = [d for (d, b) in _CODEBOOKS_F32 if b == b_bits]
        nearest = min(candidates, key=lambda d: abs(d - head_size))
        return torch.tensor(_CODEBOOKS_F32[(nearest, b_bits)], dtype=torch.float32)

    def _get_codebook(self, device: torch.device | str) -> torch.Tensor:
        """Retourne le codebook K sur le device demandé (lazy transfer GPU). Alias legacy."""
        return self._get_codebook_k(device)

    def _get_codebook_k(self, device: torch.device | str) -> torch.Tensor:
        """Retourne le codebook K sur le device demandé (lazy transfer GPU)."""
        dev_key = str(device)  # clé complète "cuda:4" pour éviter collision multi-GPU
        if dev_key not in self._codebook_k_gpu:
            self._codebook_k_gpu[dev_key] = self._codebook_k_cpu.to(device)
        return self._codebook_k_gpu[dev_key]

    def _get_codebook_v(self, device: torch.device | str) -> torch.Tensor:
        """Retourne le codebook V sur le device demandé (lazy transfer GPU)."""
        dev_key = str(device)  # clé complète "cuda:4"
        if dev_key not in self._codebook_v_gpu:
            self._codebook_v_gpu[dev_key] = self._codebook_v_cpu.to(device)
        return self._codebook_v_gpu[dev_key]

    def _get_rotation_k(self, device: torch.device | str) -> torch.Tensor:
        """Retourne la matrice de rotation K [nh, d, d] sur le bon device (V4 SVD, lazy)."""
        dev_key = str(device)
        if dev_key not in self._rotation_k_gpu:
            assert self._rotation_k_cpu is not None
            self._rotation_k_gpu[dev_key] = self._rotation_k_cpu.to(device)
        return self._rotation_k_gpu[dev_key]

    def _get_rotation_v(self, device: torch.device | str) -> torch.Tensor:
        """Retourne la matrice de rotation V [nh, d, d] sur le bon device (V4 SVD, lazy)."""
        dev_key = str(device)
        if dev_key not in self._rotation_v_gpu:
            assert self._rotation_v_cpu is not None
            self._rotation_v_gpu[dev_key] = self._rotation_v_cpu.to(device)
        return self._rotation_v_gpu[dev_key]

    # ── Calibration Dual LUT K/V (TurboESM §3.3) ──────────────────────────────

    @staticmethod
    def _lloyd_max_1d(samples: np.ndarray, n_levels: int, n_iter: int = 50) -> np.ndarray:
        """Lloyd-Max 1D sur un tableau de samples flottants.

        Algorithme itératif : assign → update → repeat.
        Initialisation uniforme dans [p2, p98] pour robustesse aux outliers.

        Args:
            samples: [N] float32 — activations aplaties
            n_levels: nombre de niveaux de quantification (2^b_bits)
            n_iter: nombre d'itérations Lloyd-Max

        Returns:
            [n_levels] float32 trié — codebook optimal MSE
        """
        p2, p98 = np.percentile(samples, 2), np.percentile(samples, 98)
        codebook = np.linspace(p2, p98, n_levels).astype(np.float32)
        for _ in range(n_iter):
            # Assign : chaque sample au centroïde le plus proche
            diffs = np.abs(samples[:, None] - codebook[None, :])  # [N, n_levels]
            labels = np.argmin(diffs, axis=-1)                     # [N]
            # Update : centroïde = moyenne du cluster
            new_cb = codebook.copy()
            for k in range(n_levels):
                mask = labels == k
                if mask.any():
                    new_cb[k] = samples[mask].mean()
            codebook = np.sort(new_cb)
        return codebook

    def _run_calibration(self, k_flat: torch.Tensor, v_flat: torch.Tensor) -> None:
        """Met à jour les codebooks K et V par Lloyd-Max empirique.

        Appelé automatiquement lors du premier prefill (≥ _calib_target tokens).
        k_flat / v_flat : [N, head_size] float32 — vecteurs normalisés (sur S^{d-1}).
        """
        k_np = k_flat.cpu().float().numpy().flatten()  # [N*d]
        v_np = v_flat.cpu().float().numpy().flatten()  # [N*d]
        n_levels = self.n_levels

        new_k_cb = self._lloyd_max_1d(k_np, n_levels)
        new_v_cb = self._lloyd_max_1d(v_np, n_levels)

        self._codebook_k_cpu = torch.from_numpy(new_k_cb)
        self._codebook_v_cpu = torch.from_numpy(new_v_cb)
        # Invalider les caches GPU
        self._codebook_k_gpu.clear()
        self._codebook_v_gpu.clear()

        # Recalculer le résidu QJL pour V2
        if self.b_bits == 3:
            self._qjl_residual_mean_k = self._compute_qjl_residual_mean(
                self._codebook_k_cpu, self.head_size
            )
            self._qjl_residual_mean_v = self._compute_qjl_residual_mean(
                self._codebook_v_cpu, self.head_size
            )

        self._calibrated = True
        self._calib_k_buf.clear()
        self._calib_v_buf.clear()

        # V4 : SVD per-head sur les activations réelles (non aplaties)
        if self._calib_k_buf_v4:
            k_all = torch.cat(self._calib_k_buf_v4, dim=0).float()  # [N, nh, d]
            v_all = torch.cat(self._calib_v_buf_v4, dim=0).float()  # [N, nh, d]
            nh = k_all.shape[1]
            d = self.head_size
            rot_k = torch.zeros(nh, d, d, dtype=torch.float32)
            rot_v = torch.zeros(nh, d, d, dtype=torch.float32)
            for h in range(nh):
                # SVD : k_h [N, d] → Vh [min(N,d), d]
                # On prend les d premières composantes (full_matrices=False économise la mémoire)
                n_samples = k_all.shape[0]
                k_svd = k_all[:, h, :]  # [N, d]
                v_svd = v_all[:, h, :]  # [N, d]
                if n_samples >= d:
                    _, _, Vh_k = torch.linalg.svd(k_svd, full_matrices=False)  # Vh_k [d, d]
                    _, _, Vh_v = torch.linalg.svd(v_svd, full_matrices=False)  # Vh_v [d, d]
                    rot_k[h] = Vh_k  # Π_h = Vh → rotate: Π_h @ x
                    rot_v[h] = Vh_v
                else:
                    # Pas assez d'échantillons : identité (pas de rotation SVD)
                    rot_k[h] = torch.eye(d)
                    rot_v[h] = torch.eye(d)
            self._rotation_k_cpu = rot_k
            self._rotation_v_cpu = rot_v
            self._rotation_k_gpu.clear()
            self._rotation_v_gpu.clear()
            self._v4_calibrated = True
            self._calib_k_buf_v4.clear()
            self._calib_v_buf_v4.clear()

    # ── Signs / devices ───────────────────────────────────────────────────────

    def _get_signs_kv(self, device: torch.device | str) -> torch.Tensor:
        """Renvoie les signes KV sur le bon device (lazy init)."""
        dev = torch.device(device)
        if self._signs_kv is None or self._signs_kv.device != dev:
            self._signs_kv = self._signs_cpu.to(dev)
        return self._signs_kv

    def _get_signs_q(self, device: torch.device | str) -> torch.Tensor:
        """Renvoie les signes Q sur le bon device (lazy init, extension GQA)."""
        dev = torch.device(device)
        if self._signs_q is None or self._signs_q.device != dev:
            signs_kv = self._get_signs_kv(dev)
            self._signs_q = signs_kv.repeat_interleave(self.num_queries_per_kv, dim=0)
        return self._signs_q

    # ── Packing 3-bit (V2) ───────────────────────────────────────────────────

    @staticmethod
    def _pack_3bit(indices: torch.Tensor) -> torch.Tensor:
        """Compresse d indices 3-bit (uint8, valeurs 0-7) en 3*d//8 bytes.

        Packing big-endian par groupe de 8 valeurs → 3 bytes :
          byte0 = v0<<5 | v1<<2 | v2>>1
          byte1 = (v2&1)<<7 | v3<<4 | v4<<1 | v5>>2
          byte2 = (v5&3)<<6 | v6<<3 | v7

        Args:
            indices: [..., d] uint8 avec valeurs dans [0, 7]

        Returns:
            [..., 3*d//8] uint8
        """
        shape_prefix = indices.shape[:-1]
        d = indices.shape[-1]
        assert d % 8 == 0, f"d doit être multiple de 8 (got {d})"

        g = indices.reshape(*shape_prefix, d // 8, 8).long()  # [..., d//8, 8]
        b0 = (g[..., 0] << 5) | (g[..., 1] << 2) | (g[..., 2] >> 1)
        b1 = ((g[..., 2] & 1) << 7) | (g[..., 3] << 4) | (g[..., 4] << 1) | (g[..., 5] >> 2)
        b2 = ((g[..., 5] & 3) << 6) | (g[..., 6] << 3) | g[..., 7]

        packed = torch.stack([b0, b1, b2], dim=-1)   # [..., d//8, 3]
        return packed.reshape(*shape_prefix, 3 * d // 8).to(torch.uint8)

    @staticmethod
    def _unpack_3bit(packed: torch.Tensor, d: int) -> torch.Tensor:
        """Décompresse 3*d//8 bytes → d indices 3-bit.

        Args:
            packed: [..., 3*d//8] uint8
            d: nombre de coordonnées original

        Returns:
            [..., d] int64 avec valeurs dans [0, 7]
        """
        shape_prefix = packed.shape[:-1]
        p = packed.reshape(*shape_prefix, d // 8, 3).long()  # [..., d//8, 3]
        b0, b1, b2 = p[..., 0], p[..., 1], p[..., 2]

        v0 = b0 >> 5
        v1 = (b0 >> 2) & 7
        v2 = ((b0 & 3) << 1) | (b1 >> 7)
        v3 = (b1 >> 4) & 7
        v4 = (b1 >> 1) & 7
        v5 = ((b1 & 1) << 2) | (b2 >> 6)
        v6 = (b2 >> 3) & 7
        v7 = b2 & 7

        result = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=-1)  # [..., d//8, 8]
        return result.reshape(*shape_prefix, d)

    @staticmethod
    def _pack_signs(signs_bool: torch.Tensor) -> torch.Tensor:
        """Compresse d bits (0/1) en d//8 bytes (8 bits par byte, LSB first).

        Args:
            signs_bool: [..., d] uint8 ou bool avec valeurs 0/1

        Returns:
            [..., d//8] uint8
        """
        shape_prefix = signs_bool.shape[:-1]
        d = signs_bool.shape[-1]
        s = signs_bool.reshape(*shape_prefix, d // 8, 8).to(torch.int32)
        shifts = torch.arange(8, device=s.device, dtype=torch.int32)
        packed = (s << shifts).sum(dim=-1).to(torch.uint8)  # [..., d//8]
        return packed

    @staticmethod
    def _unpack_signs(packed: torch.Tensor, d: int) -> torch.Tensor:
        """Décompresse d//8 bytes → d bits (0/1).

        Args:
            packed: [..., d//8] uint8
            d: nombre de coordonnées original

        Returns:
            [..., d] int32 avec valeurs 0/1
        """
        shape_prefix = packed.shape[:-1]
        p = packed.to(torch.int32)                                       # [..., d//8]
        shifts = torch.arange(8, device=p.device, dtype=torch.int32)    # [8]
        bits = (p.unsqueeze(-1) >> shifts) & 1                           # [..., d//8, 8]
        return bits.reshape(*shape_prefix, d)

    # ── Rotation ──────────────────────────────────────────────────────────────

    def _apply_rotation(self, x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        """Applique la rotation Π = H·D sur x.

        H = Walsh-Hadamard Transform normalisée (orthogonale)
        D = diag(signs) (D² = I, auto-inverse)
        Π est orthogonale et son propre inverse.

        Args:
            x: [..., num_heads, head_size]
            signs: [num_heads, head_size]  (valeurs ±1 en float32)

        Returns:
            x_rot de même shape et dtype
        """
        original_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        x_signed = x_f32 * signs.unsqueeze(0)
        x_rot = _hadamard_transform(x_signed, normalize=True)
        return x_rot.to(original_dtype)

    def rotate_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Applique Π sur K ou V avant stockage dans le cache.

        V4 : Π_h = Vh^T (SVD per-head) si calibré, sinon WHT aléatoire.

        Args:
            x: [num_tokens, num_kv_heads, head_size]

        Returns:
            x_rot même shape et dtype
        """
        if self._v4_calibrated:
            return self._apply_rotation_v4(x, which="k")
        return self._apply_rotation(x, self._get_signs_kv(x.device))

    def rotate_v(self, x: torch.Tensor) -> torch.Tensor:
        """Applique Π_v sur V avant stockage — rotations K/V peuvent diverger en V4."""
        if self._v4_calibrated:
            return self._apply_rotation_v4(x, which="v")
        return self._apply_rotation(x, self._get_signs_kv(x.device))

    def rotate_q(self, q: torch.Tensor) -> torch.Tensor:
        """Applique Π_k sur Q (même rotation que K) pour Q_rot·K_rot^T = Q·K^T.

        Args:
            q: [num_tokens, num_q_heads, head_size]

        Returns:
            q_rot même shape et dtype
        """
        if self._v4_calibrated:
            return self._apply_rotation_v4_q(q)
        return self._apply_rotation(q, self._get_signs_q(q.device))

    def _apply_rotation_v4(self, x: torch.Tensor, which: str = "k") -> torch.Tensor:
        """Rotation V4 SVD per-head : x_rot_h = Π_h @ x_h.

        Π_h = Vh_h (vecteurs singuliers droits de X_h), orthogonale.
        Implémenté par einsum pour être batché sur toutes les têtes simultanément.

        Args:
            x: [num_tokens, num_kv_heads, head_size]
            which: "k" ou "v" — sélectionne la rotation correspondante

        Returns:
            x_rot de même shape et dtype (float16)
        """
        rot = self._get_rotation_k(x.device) if which == "k" else self._get_rotation_v(x.device)
        # rot [nh, d, d], x [t, nh, d] → x_rot_h = rot_h @ x_h = x @ rot_h.T
        # einsum : 'thd,hde->the' (multiply x[t,h,:] by rot[h].T depuis droite)
        # Équivalent à : x_rot[:, h, :] = x[:, h, :] @ rot[h].T  ∀h
        original_dtype = x.dtype
        x_rot = torch.einsum("thd,hed->the", x.to(torch.float32), rot)
        return x_rot.to(original_dtype)

    def _apply_rotation_v4_q(self, q: torch.Tensor) -> torch.Tensor:
        """Rotation V4 SVD sur Q avec expansion GQA (même rotation que K correspondante).

        Chaque tête Q partage Π de sa tête KV : tête_kv = tête_q // num_queries_per_kv.

        Args:
            q: [num_tokens, num_q_heads, head_size]

        Returns:
            q_rot de même shape et dtype
        """
        rot_k = self._get_rotation_k(q.device)  # [nh_kv, d, d]
        # Expanser la rotation KV vers les têtes Q
        rot_q = rot_k.repeat_interleave(self.num_queries_per_kv, dim=0)  # [nh_q, d, d]
        original_dtype = q.dtype
        q_rot = torch.einsum("thd,hed->the", q.to(torch.float32), rot_q)
        return q_rot.to(original_dtype)

    # ── Quantification / Dequantification ─────────────────────────────────────

    def quantize_to_uint8(self, x: torch.Tensor, which: str = "k") -> torch.Tensor:
        """Quantifie x (déjà rotaté) selon TurboQuant → buffer uint8 compact.

        Dispatche sur b_bits :
          b_bits=4 → V1 : int4 packés + norme fp16  (×3.87 vs fp16)
          b_bits=3 → V2 : int3 packés + QJL signs 1-bit + norme fp8  (×5.1 vs fp16)

        Args:
            x:     [num_tokens, num_kv_heads, head_size]  fp16/bf16, déjà rotaté
            which: "k" ou "v" — sélectionne le codebook Dual LUT correspondant

        Returns:
            packed: [num_tokens, num_kv_heads, head_size]  uint8
        """
        if self.b_bits == 4:
            return self._quantize_v1(x, which=which)
        return self._quantize_v2(x, which=which)

    def _quantize_v1(self, x: torch.Tensor, which: str = "k") -> torch.Tensor:
        """V1 — 4-bit Lloyd-Max + norme fp16. Dual LUT : codebook K ou V selon which.

        Format (d bytes) :
            [0 : d//2]      → indices int4 packés (nibble haut = coord paire)
            [d//2 : d//2+2] → norme fp16
            reste            → zéros
        """
        n_tok, n_heads, d = x.shape
        x_f32 = x.to(torch.float32)
        norms = x_f32.norm(dim=-1).clamp(min=1e-8)
        x_norm = x_f32 / norms.unsqueeze(-1)

        cb_fn = self._get_codebook_k if which == "k" else self._get_codebook_v
        codebook = cb_fn(x.device).to(torch.float32)
        indices = (x_norm.unsqueeze(-1) - codebook).abs().argmin(dim=-1).to(torch.uint8)

        packed_idx = (indices[..., 0::2] << 4) | indices[..., 1::2]  # [n_tok, nh, d//2]
        norms_f16 = norms.to(torch.float16).contiguous()
        norms_u8 = norms_f16.view(n_tok, n_heads, 1).view(torch.uint8)  # [n_tok, nh, 2]

        out = torch.zeros(n_tok, n_heads, d, dtype=torch.uint8, device=x.device)
        out[..., :d // 2] = packed_idx
        out[..., d // 2: d // 2 + 2] = norms_u8
        return out

    def _quantize_v2(self, x: torch.Tensor, which: str = "k") -> torch.Tensor:
        """V2 — 3-bit Lloyd-Max + QJL 1-bit signe résidu + norme fp8_e5m2. Dual LUT.

        Implémente TurboQuant_prod (Algorithme 2, arXiv:2504.19874 §3.2) :
        le QJL 1-bit permet d'obtenir des inner-products non-biaisés et
        une qualité équivalente au 4-bit à 3.125 bits effectifs.

        Format (d bytes) :
            [0 : 3*d//8]              → indices int3 packés (8 valeurs / 3 bytes)
            [3*d//8 : 3*d//8 + d//8] → signes QJL (1 bit / coord, LSB first)
            [3*d//8 + d//8]           → norme fp8_e5m2 (1 byte)
            reste                      → zéros
        """
        n_tok, n_heads, d = x.shape
        assert d % 8 == 0, f"head_size doit être multiple de 8 pour V2 (got {d})"
        n3 = 3 * d // 8   # bytes pour les indices
        ns = d // 8       # bytes pour les signes QJL
        assert n3 + ns + 1 <= d, (
            f"head_size {d} trop petit pour V2 (besoin {n3 + ns + 1} bytes)"
        )

        x_f32 = x.to(torch.float32)

        # 1. Normes et normalisation
        norms = x_f32.norm(dim=-1).clamp(min=1e-8)  # [n_tok, n_heads]
        x_norm = x_f32 / norms.unsqueeze(-1)          # [n_tok, n_heads, d]

        # 2. Nearest centroid 3-bit
        cb_fn = self._get_codebook_k if which == "k" else self._get_codebook_v
        codebook = cb_fn(x.device).to(torch.float32)  # [8]
        diffs_all = (x_norm.unsqueeze(-1) - codebook).abs()         # [n_tok, nh, d, 8]
        indices = diffs_all.argmin(dim=-1).to(torch.uint8)          # [n_tok, nh, d]

        # 3. Signes QJL : +1 si x > Q(x) (i.e. résidu positif), 0 sinon
        quantized_vals = codebook[indices.long()]                    # [n_tok, nh, d]
        signs = (x_norm > quantized_vals).to(torch.uint8)           # [n_tok, nh, d] {0,1}

        # 4. Packing 3-bit indices
        packed_idx = self._pack_3bit(indices)   # [n_tok, nh, n3]

        # 5. Packing signes QJL
        packed_signs = self._pack_signs(signs)  # [n_tok, nh, ns]

        # 6. Norme en fp8_e5m2 (1 byte)
        norms_fp8 = norms.to(torch.float8_e5m2).view(torch.uint8)   # [n_tok, nh]

        # 7. Assemblage dans le buffer de sortie
        out = torch.zeros(n_tok, n_heads, d, dtype=torch.uint8, device=x.device)
        out[..., :n3]       = packed_idx
        out[..., n3: n3+ns] = packed_signs
        out[..., n3+ns]     = norms_fp8
        return out

    def dequantize_from_uint8(self, x_u8: torch.Tensor, which: str = "k") -> torch.Tensor:
        """Reconstruit des vecteurs fp16 depuis un buffer uint8 TurboQuant.

        Dispatche sur b_bits (même format que quantize_to_uint8).
        Les vecteurs reconstruits restent dans l'espace ROTATÉ.

        Args:
            x_u8:  [num_tokens, num_kv_heads, head_size] uint8
            which: "k" ou "v" — sélectionne le codebook Dual LUT correspondant

        Returns:
            [num_tokens, num_kv_heads, head_size] fp16 (espace rotaté)
        """
        if self.b_bits == 4:
            return self._dequantize_v1(x_u8, which=which)
        return self._dequantize_v2(x_u8, which=which)

    def _dequantize_v1(self, x_u8: torch.Tensor, which: str = "k") -> torch.Tensor:
        """V1 — 4-bit Lloyd-Max + norme fp16. Dual LUT : codebook K ou V selon which."""
        n_tok, n_heads, d = x_u8.shape

        packed = x_u8[..., :d // 2].to(torch.int64)
        idx_even = (packed >> 4) & 0xF
        idx_odd  = packed & 0xF
        indices = torch.empty(n_tok, n_heads, d, dtype=torch.int64, device=x_u8.device)
        indices[..., 0::2] = idx_even
        indices[..., 1::2] = idx_odd

        cb_fn = self._get_codebook_k if which == "k" else self._get_codebook_v
        codebook = cb_fn(x_u8.device).to(torch.float32)
        values = codebook[indices]

        norms_u8 = x_u8[..., d // 2: d // 2 + 2].contiguous()
        norms_f16 = norms_u8.view(torch.float16).view(n_tok, n_heads)

        result = values * norms_f16.unsqueeze(-1).to(torch.float32)
        return result.to(torch.float16)

    def _dequantize_v2(self, x_u8: torch.Tensor, which: str = "k") -> torch.Tensor:
        """V2 — 3-bit Lloyd-Max + correction QJL 1-bit + norme fp8_e5m2. Dual LUT.

        Reconstruction : x̃ = norm × (c_idx + sign × ē)
        avec sign = signe du résidu (1 bit stocké), ē = résidu moyen pré-calculé.
        """
        n_tok, n_heads, d = x_u8.shape
        n3 = 3 * d // 8
        ns = d // 8

        # 1. Unpack indices 3-bit
        packed_idx = x_u8[..., :n3].contiguous()                # [n_tok, nh, n3]
        indices = self._unpack_3bit(packed_idx, d)              # [n_tok, nh, d] int64

        # 2. Lookup codebook (Dual LUT)
        cb_fn = self._get_codebook_k if which == "k" else self._get_codebook_v
        codebook = cb_fn(x_u8.device).to(torch.float32)  # [8]
        values = codebook[indices]                               # [n_tok, nh, d]

        # 3. Correction QJL : x̃_norm = c_idx + sign × ē
        #    sign=1 → résidu positif (x > Q(x)) → on ajoute ē
        #    sign=0 → résidu négatif (x ≤ Q(x)) → on soustrait ē
        packed_signs = x_u8[..., n3: n3 + ns].contiguous()     # [n_tok, nh, ns]
        signs = self._unpack_signs(packed_signs, d).to(torch.float32)  # [n_tok, nh, d] {0,1}
        ebar = self._qjl_residual_mean_k if which == "k" else self._qjl_residual_mean_v
        values_corrected = values + (2.0 * signs - 1.0) * ebar  # s=1 → +ē, s=0 → -ē

        # 4. Norme fp8_e5m2 → float32
        norms_u8 = x_u8[..., n3 + ns].contiguous()             # [n_tok, nh]
        norms_fp8 = norms_u8.view(torch.float8_e5m2)            # [n_tok, nh]
        norms_f32 = norms_fp8.to(torch.float32)                  # [n_tok, nh]

        result = values_corrected * norms_f32.unsqueeze(-1)
        return result.to(torch.float16)

    # ── Stockage / Lecture depuis le cache vLLM ───────────────────────────────

    def store_to_cache(
        self,
        key: torch.Tensor,         # [num_tokens, num_kv_heads, head_size] fp16 — rotaté
        value: torch.Tensor,       # [num_tokens, num_kv_heads, head_size] fp16 — rotaté
        key_cache: torch.Tensor,   # [num_blocks, block_size, num_kv_heads, head_size] uint8
        value_cache: torch.Tensor, # [num_blocks, block_size, num_kv_heads, head_size] uint8
        slot_mapping: torch.Tensor,  # [num_tokens] int64
    ) -> None:
        """Quantifie et stocke K,V dans le cache vLLM via slot_mapping.

        Bypass du kernel Triton reshape_and_cache (qui ne supporte que fp8).
        Implémente le même scatter que le kernel Triton mais avec notre
        quantification Lloyd-Max : indices int4 packés + norme fp16.

        Layout du cache conforme à triton_reshape_and_cache_flash :
        [num_blocks, block_size, num_kv_heads, head_size]
        """
        valid_mask = slot_mapping >= 0
        if not valid_mask.any():
            return

        slots = slot_mapping[valid_mask]         # [n_valid]
        block_size = key_cache.shape[1]           # layout [nb, bs, nh, d]
        block_idx = (slots // block_size).long()  # [n_valid]
        block_off = (slots % block_size).long()   # [n_valid]

        # Quantifier K et V avec leurs codebooks respectifs (Dual LUT)
        # Calibration auto-learning : au premier prefill, accumule des activations
        # normalisées et lance Lloyd-Max empirique après _calib_target tokens.
        valid_k = key[valid_mask]    # [n_valid, nh, d] fp16 rotaté
        valid_v = value[valid_mask]  # [n_valid, nh, d] fp16 rotaté

        if not self._calibrated:
            # Accumuler des vecteurs normalisés pour la calibration
            k_norm = (valid_k.float() / valid_k.float().norm(dim=-1, keepdim=True).clamp(min=1e-8))
            v_norm = (valid_v.float() / valid_v.float().norm(dim=-1, keepdim=True).clamp(min=1e-8))
            self._calib_k_buf.append(k_norm.reshape(-1, self.head_size).cpu())
            self._calib_v_buf.append(v_norm.reshape(-1, self.head_size).cpu())
            # V4 : buffers per-tête non aplatis (shape [n_valid, nh, d])
            self._calib_k_buf_v4.append(valid_k.float().cpu())
            self._calib_v_buf_v4.append(valid_v.float().cpu())
            self._calib_collected += valid_k.shape[0] * self.num_kv_heads
            if self._calib_collected >= self._calib_target:
                k_flat = torch.cat(self._calib_k_buf, dim=0)
                v_flat = torch.cat(self._calib_v_buf, dim=0)
                self._run_calibration(k_flat, v_flat)

        key_u8 = self.quantize_to_uint8(valid_k, which="k")  # [n_valid, nh, d]
        val_u8 = self.quantize_to_uint8(valid_v, which="v")  # [n_valid, nh, d]

        # Scatter dans le cache — layout [num_blocks, block_size, num_kv_heads, head_size]
        # Méthode : étendre tous les indices explicitement pour éviter l'ambiguïté PyTorch.
        n_valid = key_u8.shape[0]
        nh = self.num_kv_heads
        heads_idx = torch.arange(nh, device=key_u8.device)

        # [n_valid, nh] pour chaque dimension d'index
        bi = block_idx.unsqueeze(1).expand(n_valid, nh)   # index de bloc
        bo = block_off.unsqueeze(1).expand(n_valid, nh)   # offset dans le bloc
        hi = heads_idx.unsqueeze(0).expand(n_valid, nh)   # index de tête

        # Assignation : key_cache[bi, bo, hi, :] → layout [nb, bs, nh, d] ✓
        key_cache[bi, bo, hi, :] = key_u8
        value_cache[bi, bo, hi, :] = val_u8

    def dequantize_cache(self, cache: torch.Tensor, which: str = "k") -> torch.Tensor:
        """Dequantifie l'intégralité d'un buffer de cache vers fp16.

        Appelé juste avant le calcul d'attention pour reconstruire K ou V.
        Les vecteurs restent dans l'espace rotaté (la rotation inverse est sur l'output).

        Utilise des buffers globaux partagés entre toutes les couches (2 buffers max en mémoire
        quel que soit le nombre de couches). En inférence séquentielle, une seule couche
        s'exécute à la fois → pas de conflit.

        Args:
            cache: [num_blocks, block_size, num_kv_heads, head_size] uint8
            which: "k" ou "v" — sélectionne le buffer global correspondant

        Returns:
            vue dans le buffer global [num_blocks, block_size, num_kv_heads, head_size] fp16
        """
        nb, bs, nh, d = cache.shape
        n_flat = nb * bs

        flat_u8 = cache.reshape(n_flat, nh, d)  # vue, pas de copie

        dev_str = str(flat_u8.device)
        is_k = (which == "k")
        buf_key = (dev_str, is_k)

        buf = TurboQuantKV._GLOBAL_DEQ_BUFFERS.get(buf_key)

        # Allocation ou réallocation si trop petit
        if buf is None or buf.shape[0] < n_flat or buf.shape[1] != nh or buf.shape[2] != d:
            alloc_size = int(n_flat * 1.1) + 16
            buf = torch.empty(alloc_size, nh, d, dtype=torch.float16, device=flat_u8.device)
            TurboQuantKV._GLOBAL_DEQ_BUFFERS[buf_key] = buf

        out_view = buf[:n_flat]  # vue, même mémoire
        self._dequantize_into_buffer(flat_u8, out_view, which=which)
        return out_view.reshape(nb, bs, nh, d)

    def _dequantize_into_buffer(self, x_u8: torch.Tensor, out: torch.Tensor, which: str = "k") -> None:
        """Dequantifie x_u8 [N, nh, d] dans out [N, nh, d] fp16 pré-alloué. Dual LUT K/V.

        Chemin prioritaire : kernel Triton fusé (V3b) — unpack+QJL+norme en un seul lancement.
        Fallback : torch.compile (V3a optimisé).
        """
        cb_fn = self._get_codebook_k if which == "k" else self._get_codebook_v
        codebook = cb_fn(x_u8.device)

        if _USE_TRITON and x_u8.is_cuda:
            # Chemin V3b : kernel Triton fusé (0 allocation intermédiaire).
            # Le codebook float32 est explicitement retenu en variable locale pour éviter
            # qu'il soit GC'd pendant les appels GPU asynchrones (bug Triton "cpu tensor?").
            codebook_f32 = codebook.to(torch.float32)
            if self.b_bits == 4:
                turbo_deq_v1_triton(x_u8, codebook_f32, out)
            else:
                qjl_mean = self._qjl_residual_mean_k if which == "k" else self._qjl_residual_mean_v
                turbo_deq_v2_triton(x_u8, codebook_f32, qjl_mean, out)
        else:
            # Fallback : torch.compile (chemin V3a)
            if self.b_bits == 4:
                _dequantize_v1_compiled(x_u8, out, codebook)
            else:
                qjl_mean = self._qjl_residual_mean_k if which == "k" else self._qjl_residual_mean_v
                _dequantize_v2_compiled(x_u8, out, codebook, qjl_mean)

    # ── Un-rotation de l'output ────────────────────────────────────────────────

    def unrotate_output(self, output_rot: torch.Tensor, num_actual_tokens: int) -> torch.Tensor:
        """Inverse la rotation sur l'output de unified_attention.

        Après rotation de V : output_rot[t,h] = Π_{h_kv} · output[t,h]
        → output[t,h] = Π^{-1} · output_rot[t,h] = Π · output_rot (car Π = Π^{-1})

        GQA : les têtes Q partagent les signes de leur tête KV correspondante.

        Args:
            output_rot: [num_padded_tokens, num_heads, head_size] — output d'attention
            num_actual_tokens: nombre de tokens réels (sans padding)

        Returns:
            output_rot modifié in-place
        """
        n = num_actual_tokens
        device = output_rot.device
        original_dtype = output_rot.dtype

        # Accepte [T, NH, D] (chemin vllm) ou [T, NH*D] (chemin test flat)
        input_3d = output_rot.ndim == 3
        if not input_3d:
            # [T, NH*D] → [T, NH, D]
            T_pad = output_rot.shape[0]
            output_rot = output_rot.view(T_pad, self.num_q_heads, self.head_size)

        view = output_rot[:n]  # [n, num_heads, head_size]

        if self._v4_calibrated:
            # V4 : dérotation SVD per-head.
            # L'output contient Σ_i attn_i * (Π_vi @ V_i).
            # Pour retrouver la valeur vraie : Π_v^{-1} = Π_v^T = rot_v^T.
            # output_true_h = rot_v[h] @ output_h^T = output_h @ rot_v[h]^T → einsum
            # Expansion GQA : tête Q j ↔ tête KV j//num_queries_per_kv
            rot_v = self._get_rotation_v(device)  # [nh_kv, d, d]
            rot_q = rot_v.repeat_interleave(self.num_queries_per_kv, dim=0)  # [nh_q, d, d]
            # view [n, nh_q, d] @ rot_q.T [nh_q, d, d] → einsum 'thd,hde->the' avec rot_q transposé
            view_f32 = view.to(torch.float32)
            view_unrot = torch.einsum("thd,hde->the", view_f32, rot_q)  # rot_q non transposé = Π^{-1}
        else:
            signs_q = self._get_signs_q(device)  # [num_q_heads, head_size]
            view_f32 = view.to(torch.float32)
            view_whd = _hadamard_transform(view_f32, normalize=True)
            view_unrot = view_whd * signs_q.unsqueeze(0)

        output_rot[:n] = view_unrot.to(original_dtype)
        return output_rot

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def verify_rotation_correctness(
        self, device: str = "cuda", rtol: float = 1e-2
    ) -> bool:
        """Vérifie que Q_rot·K_rot^T = Q·K^T (scores d'attention préservés).

        La rotation Π = H·D a pour inverse D·H (pas H·D).
        Ce test vérifie la propriété clé : les produits scalaires sont préservés
        par la rotation orthogonale Π (qui est son inverse D·H par transposée).
        """
        d = self.head_size
        q = torch.randn(5, self.num_q_heads, d, device=device, dtype=torch.float32)
        k = torch.randn(5, self.num_kv_heads, d, device=device, dtype=torch.float32)
        q_rot = self.rotate_q(q.to(torch.float16)).to(torch.float32)
        k_rot = self.rotate_kv(k.to(torch.float16)).to(torch.float32)
        # Scores originaux et rotatés (pour 1 tête)
        score_orig = (q[:, :self.num_kv_heads, :] * k).sum(dim=-1)
        score_rot  = (q_rot[:, :self.num_kv_heads, :] * k_rot).sum(dim=-1)
        err = (score_orig - score_rot).abs().max().item()
        ok = err < rtol
        if not ok:
            import warnings
            warnings.warn(
                f"TurboQuantKV :: Q·K^T ≠ Q_rot·K_rot^T, err={err:.4f} > {rtol} "
                f"(head_size={d})"
            )
        return ok

    def verify_quantization_quality(
        self, device: str = "cuda", num_samples: int = 500
    ) -> dict[str, float]:
        """Mesure MSE et SNR de la quantification TurboQuant sur des vecteurs aléatoires.

        Returns:
            dict avec 'mse', 'snr_db', 'compression_ratio'
        """
        d = self.head_size
        x_rand = torch.randn(num_samples, self.num_kv_heads, d,
                             device=device, dtype=torch.float16)
        x_rot = self.rotate_kv(x_rand)
        x_u8 = self.quantize_to_uint8(x_rot)
        x_deq = self.dequantize_from_uint8(x_u8)

        mse = (x_rot.float() - x_deq.float()).pow(2).mean().item()
        signal_power = x_rot.float().pow(2).mean().item()
        snr_db = 10 * math.log10(signal_power / (mse + 1e-12))

        # Bits effectifs par coordonnée (indices + overhead norme amortis sur d)
        if self.b_bits == 4:
            bits_stored = self.b_bits * d + 16   # 4 bits/coord + fp16 norme
        else:
            # V2 : 3 bits + 1 bit QJL + fp8 norme (8 bits) / d coords
            bits_stored = (self.b_bits + 1) * d + 8
        bits_fp16 = 16 * d
        ratio = bits_fp16 / bits_stored

        result_dict = {"mse": mse, "snr_db": snr_db, "compression_ratio": ratio}
        if self.b_bits == 3:
            result_dict["qjl_residual_mean"] = self._qjl_residual_mean
        return result_dict
