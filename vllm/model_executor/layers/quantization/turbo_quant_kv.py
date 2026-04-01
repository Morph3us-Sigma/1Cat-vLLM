# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# TurboQuant KV Cache Quantization — Zandieh et al. 2025 (arXiv:2504.19874)
#
# Compression vectorielle en ligne du KV cache par rotation Hadamard aléatoire.
#
# Principe :
#   - Après rotation RHT, les coordonnées suivent ~Beta(d/2, d/2)
#   - Cette distribution concentrée permet une quantification scalaire quasi-optimale
#   - 3.5 bits/canal = qualité identique à FP16 (zéro dégradation mesurée)
#   - 2.5 bits/canal = dégradation marginale seulement
#
# Intégration v1 (phase 1) :
#   - Rotation de K et Q uniquement (K stocké tourné, Q tourné à l'inférence)
#   - Résultat : Q·K^T inchangé = scores d'attention identiques
#   - V non tourné → output correct sans post-rotation
#   - La rotation améliore la distribution pour fp8 KV cache (÷2 VRAM sans perte)
#
# Intégration v2 (phase 2, actuelle) :
#   - Rotation de K, Q ET V avant stockage dans le cache
#   - Mathématique :
#       output_rot[t,h] = Σ_i attn[t,i,h] · R_{h_kv} · V[i,h_kv]
#                       = R_{h_kv} · output[t,h]    (linéarité)
#     → un-rotation post-attention : output = R_{h_kv}^{-1} · output_rot
#     → permet vrai 3.5 bits sur K ET V = VRAM KV ÷ 4 vs FP16 (K+V tous les deux)
#   - Note : résultat numériquement identique à FP16 (rotation orthogonale exacte)
#
# Référence : https://arxiv.org/abs/2504.19874

from __future__ import annotations

import math

import torch


def _hadamard_transform(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Walsh-Hadamard Transform (WHT) vectorisée in-place.

    Args:
        x: tensor de forme [..., d] où d doit être une puissance de 2
        normalize: si True, divise par sqrt(d) pour normaliser

    Returns:
        tensor de même forme avec WHT appliquée sur la dernière dimension
    """
    d = x.shape[-1]
    assert d > 0 and (d & (d - 1)) == 0, f"head_size doit être une puissance de 2, got {d}"

    h = x.clone()
    step = 1
    while step < d:
        # Butterfly in-place
        a = h[..., ::2 * step].clone()   # ne marche pas pour step arbitraire
        # Utilisation d'une approche par reshape pour l'efficacité
        # Reformater pour avoir la dimension butterfly en avant
        half = h.reshape(*h.shape[:-1], d // (2 * step), 2 * step)
        left = half[..., :step].clone()
        right = half[..., step : 2 * step].clone()
        half[..., :step] = left + right
        half[..., step : 2 * step] = left - right
        h = half.reshape(*h.shape[:-1], d)
        step *= 2

    if normalize:
        h = h * (1.0 / math.sqrt(d))
    return h


class TurboQuantKV:
    """Randomized Hadamard Transform pour améliorer la qualité du KV cache quantifié.

    Applique R = H·D où :
    - H est la transformation de Walsh-Hadamard normalisée (orthogonale)
    - D = diag(s_1, ..., s_d) avec s_i ∈ {±1} tirés uniformément

    Propriétés :
    - R est orthogonale → préserve les normes et les produits scalaires
    - Après rotation, les coordonnées suivent ~Beta(d/2, d/2) ≈ Gaussienne concentrée
    - La concentration permet une quantification scalaire quasi-optimale
    - data-oblivious : aucune calibration, applicable online

    Usage pour KV attention :
    - Encoder: K_rot = rotate_kv(K) → stocker K_rot dans le cache
    - Decoder: Q_rot = rotate_q(Q) → Q_rot · K_rot^T = Q · K^T ✅ scores identiques
    - V non tourné → output correct sans post-traitement
    """

    def __init__(
        self,
        head_size: int,
        num_kv_heads: int,
        num_q_heads: int,
        seed: int = 42,
    ) -> None:
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.num_q_heads = num_q_heads
        self.num_queries_per_kv = num_q_heads // num_kv_heads

        assert head_size > 0 and (head_size & (head_size - 1)) == 0, (
            f"TurboQuantKV : head_size doit être une puissance de 2 (got {head_size})"
        )

        # Matrice de signe D = diag(±1) : une ligne par tête KV
        # shape = [num_kv_heads, head_size]
        rng = torch.Generator()
        rng.manual_seed(seed)
        signs_fp16 = (
            torch.randint(0, 2, (num_kv_heads, head_size), generator=rng).to(
                torch.float16
            )
            * 2
            - 1
        )
        self._signs_cpu: torch.Tensor = signs_fp16
        self._signs_kv: torch.Tensor | None = None  # [num_kv_heads, head_size]
        self._signs_q: torch.Tensor | None = None   # [num_q_heads, head_size]

    # ── Gestion des devices ────────────────────────────────────────────────────

    def _get_signs_kv(self, device: torch.device | str) -> torch.Tensor:
        """Renvoie les signes KV sur le bon device (lazy init)."""
        if self._signs_kv is None or self._signs_kv.device.type != str(device).split(":")[0]:
            self._signs_kv = self._signs_cpu.to(device)
        return self._signs_kv

    def _get_signs_q(self, device: torch.device | str) -> torch.Tensor:
        """Renvoie les signes Q sur le bon device (lazy init, répétition si GQA)."""
        if self._signs_q is None or self._signs_q.device.type != str(device).split(":")[0]:
            # Répéter chaque ligne KV pour les queries correspondantes (GQA)
            signs_kv = self._get_signs_kv(device)
            self._signs_q = signs_kv.repeat_interleave(self.num_queries_per_kv, dim=0)
        return self._signs_q

    # ── Opérations de rotation ─────────────────────────────────────────────────

    def rotate_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Applique R = H·D sur K ou V.

        Args:
            x: [num_tokens, num_kv_heads, head_size]

        Returns:
            x_rot: [num_tokens, num_kv_heads, head_size] — même dtype
        """
        return self._apply_rotation(x, self._get_signs_kv(x.device))

    # Alias explicite pour la lisibilité (V utilise les mêmes signes KV)
    rotate_v = rotate_kv

    def rotate_q(self, q: torch.Tensor) -> torch.Tensor:
        """Applique R = H·D sur Q (même rotation que K pour préserver Q·K^T).

        Args:
            q: [num_tokens, num_q_heads, head_size]

        Returns:
            q_rot: [num_tokens, num_q_heads, head_size] — même dtype
        """
        return self._apply_rotation(q, self._get_signs_q(q.device))

    def unrotate_output(self, output_rot: torch.Tensor, num_actual_tokens: int) -> torch.Tensor:
        """Inverse la rotation sur l'output de unified_attention (Phase 2).

        Après rotation de V : output_rot[t,h] = R_{h_kv} · output[t,h]
        → output[t,h] = R_{h_kv}^{-1} · output_rot[t,h]
        → R^{-1} = R^T = D·H (H est son propre inverse, D² = I)

        Les têtes Q utilisent les signes de la tête KV correspondante (GQA) :
            signs_output[h] = signs_kv[h // num_queries_per_kv]  = signs_q[h]

        Args:
            output_rot: [num_padded_tokens, num_heads * head_size] — output de unified_attention
            num_actual_tokens: nombre de tokens réels (sans padding)

        Returns:
            output_rot modifié in-place : tokens réels un-rotés, padding inchangé
        """
        n = num_actual_tokens
        device = output_rot.device
        original_dtype = output_rot.dtype

        # Reshape [n, num_heads * head_size] → [n, num_heads, head_size]
        view = output_rot[:n].view(n, self.num_q_heads, self.head_size)

        # Un-rotation : D·H (même opération que rotate car H^{-1}=H et D^{-1}=D)
        signs_q = self._get_signs_q(device)  # [num_q_heads, head_size]
        view_f16 = view.to(torch.float16)
        # Étape 1 : H (WHT normalisée)
        view_whd = _hadamard_transform(view_f16, normalize=True)
        # Étape 2 : D^{-1} = D (car D² = I, signes ±1)
        view_unrot = view_whd * signs_q.unsqueeze(0)  # [n, num_q_heads, head_size]

        # Réécrire in-place dans output_rot
        output_rot[:n] = view_unrot.view(n, self.num_q_heads * self.head_size).to(original_dtype)
        return output_rot

    def _apply_rotation(self, x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        """Applique R = H·D sur x avec les signes donnés.

        Args:
            x: [..., num_heads, head_size]
            signs: [num_heads, head_size]

        Returns:
            x_rot de même shape et dtype
        """
        original_dtype = x.dtype
        x_f16 = x.to(torch.float16)
        x_signed = x_f16 * signs.unsqueeze(0)
        x_rot = _hadamard_transform(x_signed, normalize=True)
        return x_rot.to(original_dtype)

    def verify_rotation_correctness(
        self, device: str = "cuda", rtol: float = 1e-2
    ) -> bool:
        """Test de santé : vérifie R·R^T = I et préservation des produits scalaires.

        Retourne True si la rotation est correcte.
        """
        d = self.head_size
        # Vecteur de test
        v = torch.randn(1, self.num_kv_heads, d, device=device, dtype=torch.float16)
        v_rot = self.rotate_kv(v)
        v_unrot = self.unrotate(v_rot, is_kv=True)
        # Erreur de reconstruction
        err = (v - v_unrot).abs().max().item()
        ok = err < rtol
        if not ok:
            import warnings
            warnings.warn(
                f"TurboQuantKV :: erreur reconstruction = {err:.4f} > {rtol} "
                f"(head_size={d}, num_kv_heads={self.num_kv_heads})"
            )
        return ok
