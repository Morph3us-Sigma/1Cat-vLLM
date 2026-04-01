# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Kernel Triton fusé pour TurboQuant KV Cache — V3b
#
# Fusionne : unpack int3 + correction QJL 1-bit + multiplication norme fp8_e5m2 → fp16
# en un seul kernel GPU, sans allocation intermédiaire.
#
# Gain mesuré (TurboESM arXiv:2603.26110) : +1.96× decode KV fetch vs chemin PyTorch 2-étapes.
#
# Architecture :
#  - Un thread Triton par token × tête KV
#  - Charge les bytes uint8 compactés, déballe en place
#  - Calcule les valeurs fp32 via LUT codebook constante (partagée en SMEM)
#  - Applique la norme fp8 et écrit fp16 directement
#
# Format V2 (d=128) :
#   [0:48]   → 48 bytes = 128 × 3-bit indices (big-endian 8 valeurs / 3 bytes)
#   [48:64]  → 16 bytes = 128 × 1-bit signes QJL (8/byte, LSB first)
#   [64]     →  1 byte  = norme fp8_e5m2
#   [65:128] → padding

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ── Kernel Triton V2 (3-bit + QJL + fp8 → fp16) ──────────────────────────────

if _TRITON_AVAILABLE:

    @triton.jit
    def _turbo_deq_v2_kernel(
        # Entrée : cache uint8 [N, NH, D]
        x_ptr,
        stride_x_n, stride_x_h, stride_x_d,
        # Sortie : fp16 [N, NH, D]
        out_ptr,
        stride_out_n, stride_out_h, stride_out_d,
        # Codebook float32 [n_levels] — pointeur vers constante
        cb_ptr,
        # QJL résidu moyen ē (scalaire)
        ebar: tl.constexpr,
        # Dimensions
        N: tl.constexpr,    # num_tokens
        NH: tl.constexpr,   # num_kv_heads
        D: tl.constexpr,    # head_size (doit être multiple de 8)
        # Format dérivé
        N3: tl.constexpr,   # 3*D//8 = nombre de bytes indices 3-bit
        NS: tl.constexpr,   # D//8   = nombre de bytes signes QJL
        BLOCK_D: tl.constexpr,  # = D (traite une tête entière par thread-block)
    ):
        """Dequantifie V2 : uint8 → fp16 en un seul kernel.

        Grid : (N * NH,) — un thread-block par token × tête.
        Chaque thread-block traite BLOCK_D coordonnées (= D).
        """
        # Identité du thread-block
        prog_id = tl.program_id(0)
        tok_id = prog_id // NH
        head_id = prog_id % NH

        # Charger les coordonnées de ce token × tête
        base_x   = tok_id * stride_x_n + head_id * stride_x_h
        base_out = tok_id * stride_out_n + head_id * stride_out_h

        # Charger le codebook en registres (8 niveaux max)
        n_levels = 8  # V2 = 3-bit → 8 niveaux
        cb = tl.load(cb_ptr + tl.arange(0, 8))  # [8] float32

        # ── Chargement des indices 3-bit (N3 bytes) ──
        # Chaque groupe de 3 bytes encode 8 valeurs 3-bit
        # On traite les D coordonnées par groupes de 8 (= BLOCK_D // 8 groupes)
        offs_coord = tl.arange(0, BLOCK_D)  # [D]

        # Indices des groupes de 3 bytes
        group_id   = offs_coord // 8         # [D] — groupe d'appartenance
        local_pos  = offs_coord % 8          # [D] — position dans le groupe

        # Les 3 bytes du groupe : positions byte0, byte1, byte2 dans x
        byte0_offs = group_id * 3            # [D]
        byte1_offs = group_id * 3 + 1        # [D]
        byte2_offs = group_id * 3 + 2        # [D]

        b0 = tl.load(x_ptr + base_x + byte0_offs, mask=byte0_offs < N3).to(tl.int32)
        b1 = tl.load(x_ptr + base_x + byte1_offs, mask=byte1_offs < N3).to(tl.int32)
        b2 = tl.load(x_ptr + base_x + byte2_offs, mask=byte2_offs < N3).to(tl.int32)

        # Décodage big-endian des 8 valeurs 3-bit depuis 3 bytes :
        # v0 = b0>>5, v1 = (b0>>2)&7, v2 = ((b0&3)<<1)|(b1>>7)
        # v3 = (b1>>4)&7, v4 = (b1>>1)&7, v5 = ((b1&1)<<2)|(b2>>6)
        # v6 = (b2>>3)&7, v7 = b2&7
        v0 = b0 >> 5
        v1 = (b0 >> 2) & 7
        v2 = ((b0 & 3) << 1) | (b1 >> 7)
        v3 = (b1 >> 4) & 7
        v4 = (b1 >> 1) & 7
        v5 = ((b1 & 1) << 2) | (b2 >> 6)
        v6 = (b2 >> 3) & 7
        v7 = b2 & 7

        # Sélectionner v[local_pos] pour chaque coordonnée
        idx = (
            tl.where(local_pos == 0, v0,
            tl.where(local_pos == 1, v1,
            tl.where(local_pos == 2, v2,
            tl.where(local_pos == 3, v3,
            tl.where(local_pos == 4, v4,
            tl.where(local_pos == 5, v5,
            tl.where(local_pos == 6, v6, v7)))))))
        )  # [D] int32, 0-7

        # Lookup codebook [D] → float32
        values = tl.gather(cb, idx, 0)  # [D] float32

        # ── Charger les signes QJL (NS bytes) ──
        sign_byte_id  = offs_coord // 8     # [D] — même indexation que les groupes
        sign_bit_pos  = offs_coord % 8      # [D] — bit de position (LSB first)
        sign_byte_val = tl.load(
            x_ptr + base_x + N3 + sign_byte_id,
            mask=sign_byte_id < NS
        ).to(tl.int32)
        signs = (sign_byte_val >> sign_bit_pos) & 1  # [D] {0, 1}

        # Correction QJL : x̃ = v + (2*s - 1) * ē
        values_corr = values + (2.0 * signs.to(tl.float32) - 1.0) * ebar

        # ── Charger la norme fp8_e5m2 ──
        norm_u8 = tl.load(x_ptr + base_x + N3 + NS).to(tl.uint8)
        # Conversion fp8_e5m2 → float32 :
        # fp8_e5m2 : signe(1) exposant(5) mantisse(2)
        # Formula : val = (-1)^s × 2^(e-15) × (1 + m/4) pour e≠0
        sign_bit  = (norm_u8 >> 7) & 1
        exp_bits  = (norm_u8 >> 2) & 0x1F
        mant_bits = norm_u8 & 0x3
        # Cas normal (e≠0, ≠31)
        exp_val  = exp_bits.to(tl.float32) - 15.0
        mant_val = 1.0 + mant_bits.to(tl.float32) * 0.25
        norm_f32 = tl.where(
            exp_bits == 0,
            mant_bits.to(tl.float32) * 0.25 * (2.0 ** (-14.0)),
            mant_val * tl.exp2(exp_val)
        )
        norm_f32 = tl.where(sign_bit == 1, -norm_f32, norm_f32)

        # ── Écriture fp16 ──
        result_f16 = (values_corr * norm_f32).to(tl.float16)
        tl.store(out_ptr + base_out + offs_coord, result_f16)


    @triton.jit
    def _turbo_deq_v1_kernel(
        x_ptr,
        stride_x_n, stride_x_h, stride_x_d,
        out_ptr,
        stride_out_n, stride_out_h, stride_out_d,
        cb_ptr,
        N: tl.constexpr,
        NH: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Dequantifie V1 : int4 packés + norme fp16 → fp16."""
        prog_id = tl.program_id(0)
        tok_id  = prog_id // NH
        head_id = prog_id % NH

        base_x   = tok_id * stride_x_n + head_id * stride_x_h
        base_out = tok_id * stride_out_n + head_id * stride_out_h

        # Codebook 4-bit (16 niveaux)
        cb = tl.load(cb_ptr + tl.arange(0, 16))

        offs_coord = tl.arange(0, BLOCK_D)   # [D]
        half_d = D // 2

        # Charger les nibbles packés : D/2 bytes → D indices
        packed_offs = offs_coord // 2         # [D] → byte index
        packed_byte = tl.load(
            x_ptr + base_x + packed_offs,
            mask=packed_offs < half_d
        ).to(tl.int32)

        # Nibble haut = coord paire, nibble bas = coord impaire
        is_even = (offs_coord % 2) == 0
        idx = tl.where(is_even, (packed_byte >> 4) & 0xF, packed_byte & 0xF)

        values = tl.gather(cb, idx, 0)  # [D] float32

        # Charger la norme fp16 (2 bytes)
        norm_lo = tl.load(x_ptr + base_x + half_d).to(tl.uint16)
        norm_hi = tl.load(x_ptr + base_x + half_d + 1).to(tl.uint16)
        norm_bits = (norm_hi.to(tl.uint32) << 8) | norm_lo.to(tl.uint32)
        # fp16 → float32 : signe(1) exp(5) mant(10)
        s = (norm_bits >> 15) & 1
        e = (norm_bits >> 10) & 0x1F
        m = norm_bits & 0x3FF
        norm_f32 = tl.where(
            e == 0,
            m.to(tl.float32) * (2.0 ** (-24.0)),
            (1.0 + m.to(tl.float32) * (1.0 / 1024.0)) * tl.exp2(e.to(tl.float32) - 15.0)
        )
        norm_f32 = tl.where(s == 1, -norm_f32, norm_f32)

        result_f16 = (values * norm_f32).to(tl.float16)
        tl.store(out_ptr + base_out + offs_coord, result_f16)


# ── Interface Python ──────────────────────────────────────────────────────────

def turbo_deq_v2_triton(
    x_u8: torch.Tensor,         # [N, NH, D] uint8
    codebook: torch.Tensor,     # [8] float32 sur GPU
    ebar: float,                # résidu QJL
    out: torch.Tensor,          # [N, NH, D] fp16 pré-alloué
) -> None:
    """Dequantifie V2 via kernel Triton fusé (unpack3bit + QJL + norme → fp16).

    Grid : (N*NH,) — un programme par token × tête.
    Plus rapide que le chemin PyTorch 2-étapes car élimine les allocations intermédiaires.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton non disponible — utiliser _dequantize_v2_inplace")

    N, NH, D = x_u8.shape
    assert D % 8 == 0
    n3 = 3 * D // 8
    ns = D // 8

    grid = (N * NH,)
    _turbo_deq_v2_kernel[grid](
        x_u8, x_u8.stride(0), x_u8.stride(1), x_u8.stride(2),
        out, out.stride(0), out.stride(1), out.stride(2),
        codebook,
        ebar,
        N=N, NH=NH, D=D,
        N3=n3, NS=ns,
        BLOCK_D=D,
        num_warps=4,  # type: ignore[call-arg]
        num_stages=2,  # type: ignore[call-arg]
    )


def turbo_deq_v1_triton(
    x_u8: torch.Tensor,         # [N, NH, D] uint8
    codebook: torch.Tensor,     # [16] float32 sur GPU
    out: torch.Tensor,          # [N, NH, D] fp16 pré-alloué
) -> None:
    """Dequantifie V1 via kernel Triton fusé (int4 + norme fp16 → fp16)."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton non disponible — utiliser _dequantize_v1_inplace")

    N, NH, D = x_u8.shape
    grid = (N * NH,)
    _turbo_deq_v1_kernel[grid](
        x_u8, x_u8.stride(0), x_u8.stride(1), x_u8.stride(2),
        out, out.stride(0), out.stride(1), out.stride(2),
        codebook,
        N=N, NH=NH, D=D,
        BLOCK_D=D,
        num_warps=4,  # type: ignore[call-arg]
        num_stages=2,  # type: ignore[call-arg]
    )


def is_triton_available() -> bool:
    return _TRITON_AVAILABLE
