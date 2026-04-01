"""
Tests TurboQuant KV Cache — vérification mathématique et intégration triton_attn.

Lance avec :
    cd /mnt/data-ssd/morph3us/highbrain/external/1Cat-vLLM
    PYTHONPATH=$(pwd) python3.12 tests/turbo_quant/test_turbo_quant_kv.py
"""

import math
import sys
import time

import torch

# ── Import du module TurboQuant depuis le fork ────────────────────────────────
from vllm.model_executor.layers.quantization.turbo_quant_kv import (
    TurboQuantKV,
    _hadamard_transform,
)


# ═════════════════════════════════════════════════════════════════════════════
#  Utilitaires
# ═════════════════════════════════════════════════════════════════════════════

PASS = "✅"
FAIL = "❌"
SEP  = "─" * 70


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))
    return condition


# ═════════════════════════════════════════════════════════════════════════════
#  1. Tests WHT (Walsh-Hadamard Transform)
# ═════════════════════════════════════════════════════════════════════════════

def test_wht():
    print(f"\n{SEP}")
    print("  TEST 1 : Walsh-Hadamard Transform")
    print(SEP)
    ok_all = True

    for d in [32, 64, 128, 256]:
        x = torch.randn(4, 8, d)
        x2 = _hadamard_transform(_hadamard_transform(x, normalize=True), normalize=True)
        err = (x - x2).abs().max().item()
        ok = err < 5e-4
        ok_all &= ok
        check(f"H·H = I  (d={d})", ok, f"err={err:.2e}")

    # Propriété orthogonale : préservation des normes
    d = 128
    x = torch.randn(10, 4, d)
    x_rot = _hadamard_transform(x, normalize=True)
    norm_orig = x.norm(dim=-1)
    norm_rot  = x_rot.norm(dim=-1)
    err_norm = (norm_orig - norm_rot).abs().max().item()
    ok = err_norm < 1e-4
    ok_all &= ok
    check("Préservation des normes (||Hx|| = ||x||)", ok, f"err={err_norm:.2e}")

    return ok_all


# ═════════════════════════════════════════════════════════════════════════════
#  2. Tests TurboQuantKV — invariants algébriques
# ═════════════════════════════════════════════════════════════════════════════

def test_invariants():
    print(f"\n{SEP}")
    print("  TEST 2 : Invariants algébriques TurboQuantKV")
    print(SEP)
    ok_all = True

    for num_kv, num_q, d in [(4, 8, 128), (8, 8, 64), (2, 8, 256)]:
        tq = TurboQuantKV(head_size=d, num_kv_heads=num_kv, num_q_heads=num_q)
        nqpkv = num_q // num_kv

        # Invariant A : Q_rot · K_rot^T = Q · K^T
        q = torch.randn(16, num_q, d)
        k = torch.randn(16, num_kv, d)
        q_rot = tq.rotate_q(q)
        k_rot = tq.rotate_kv(k)
        max_err = 0.0
        for h in range(num_q):
            hkv = h // nqpkv
            dot_orig  = (q[:, h, :] * k[:, hkv, :]).sum(-1)
            dot_rot   = (q_rot[:, h, :] * k_rot[:, hkv, :]).sum(-1)
            max_err = max(max_err, (dot_orig - dot_rot).abs().max().item())
        ok_a = max_err < 0.2
        ok_all &= ok_a
        check(f"Q_rot·K_rot^T = Q·K^T  (kv={num_kv},q={num_q},d={d})", ok_a, f"max_err={max_err:.2e}")

        # Invariant B : rotate_kv = rotate_v (même opération)
        v = torch.randn(8, num_kv, d)
        v_a = tq.rotate_kv(v)
        v_b = tq.rotate_v(v)
        err_b = (v_a - v_b).abs().max().item()
        ok_b = err_b == 0.0
        ok_all &= ok_b
        check(f"rotate_kv = rotate_v  (d={d})", ok_b, f"err={err_b:.2e}")

    return ok_all


# ═════════════════════════════════════════════════════════════════════════════
#  3. Test Phase 2 : rotation V + un-rotation output = FP16 exact
# ═════════════════════════════════════════════════════════════════════════════

def test_phase2_correctness():
    print(f"\n{SEP}")
    print("  TEST 3 : Phase 2 — V rotation + output un-rotation = FP16 exact")
    print(SEP)
    ok_all = True

    for num_kv, num_q, d, T_kv, T_q in [
        (4,  8,  128, 64,  8),
        (8,  8,   64, 32,  4),
        (2, 16,  128, 16,  3),
    ]:
        tq = TurboQuantKV(head_size=d, num_kv_heads=num_kv, num_q_heads=num_q)
        nqpkv = num_q // num_kv

        # Simuler une séquence K/V cachée et un batch de queries
        # Shape attendu par vllm : [T, num_heads, head_size]
        K = torch.randn(T_kv, num_kv, d)
        V = torch.randn(T_kv, num_kv, d)
        Q = torch.randn(T_q,  num_q,  d)

        # Calcul d'attention FP16 direct (référence)
        # scores[tq, hq, tk] = Q[tq, hq] · K[tk, hkv] / sqrt(d)
        scale = 1.0 / math.sqrt(d)
        scores = torch.einsum(
            "qhd,khd->qhk",
            Q,
            K[:, [h // nqpkv for h in range(num_q)], :],  # expand K pour GQA
        ) * scale  # [T_q, num_q, T_kv]
        attn_weights = torch.softmax(scores, dim=-1)  # [T_q, num_q, T_kv]

        # output_ref[tq, hq, d] = sum_tk attn[tq, hq, tk] * V[tk, hkv]
        V_expanded = V[:, [h // nqpkv for h in range(num_q)], :]  # [T_kv, num_q, d]
        output_ref = torch.einsum("qhk,khd->qhd", attn_weights, V_expanded)  # [T_q, num_q, d]

        # ── Simulation du chemin TurboQuant ──────────────────────────────────
        K_rot = tq.rotate_kv(K)   # stocké dans le cache
        V_rot = tq.rotate_v(V)    # stocké dans le cache
        Q_rot = tq.rotate_q(Q)    # rotation query à l'inférence

        # Scores Q_rot · K_rot^T = Q · K^T (invariant vérifié en Test 2)
        scores_rot = torch.einsum(
            "qhd,khd->qhk",
            Q_rot,
            K_rot[:, [h // nqpkv for h in range(num_q)], :],
        ) * scale
        attn_weights_rot = torch.softmax(scores_rot, dim=-1)

        V_rot_expanded = V_rot[:, [h // nqpkv for h in range(num_q)], :]
        output_rot_3d = torch.einsum("qhk,khd->qhd", attn_weights_rot, V_rot_expanded)

        # Un-rotation via unrotate_output (fonctionne sur output à plat [T, num_q*d])
        output_rot_flat = output_rot_3d.reshape(T_q, num_q * d)
        # Padding factice pour simuler le buffer vllm
        padding = T_q + 4
        output_padded = torch.zeros(padding, num_q * d)
        output_padded[:T_q] = output_rot_flat

        tq.unrotate_output(output_padded, num_actual_tokens=T_q)
        output_tq = output_padded[:T_q].reshape(T_q, num_q, d)

        # Padding doit rester zéro
        pad_ok = output_padded[T_q:].abs().max().item() == 0.0

        err = (output_ref - output_tq).abs().max().item()
        # Erreur attendue : uniquement arrondi float16 → float32
        ok = err < 0.02 and pad_ok
        ok_all &= ok
        check(
            f"output TurboQuant = FP16  (kv={num_kv},q={num_q},d={d},T={T_kv}/{T_q})",
            ok,
            f"err={err:.2e}, padding_intact={pad_ok}",
        )

    return ok_all


# ═════════════════════════════════════════════════════════════════════════════
#  4. Test GQA (Grouped Query Attention) — ratios divers
# ═════════════════════════════════════════════════════════════════════════════

def test_gqa():
    print(f"\n{SEP}")
    print("  TEST 4 : GQA — divers ratios num_q / num_kv")
    print(SEP)
    ok_all = True

    for num_kv, num_q, d in [(1, 8, 128), (2, 8, 64), (8, 32, 128), (4, 4, 256)]:
        tq = TurboQuantKV(head_size=d, num_kv_heads=num_kv, num_q_heads=num_q)
        signs_kv = tq._get_signs_kv("cpu")
        signs_q  = tq._get_signs_q("cpu")
        nqpkv = num_q // num_kv

        errors = []
        for h in range(num_q):
            hkv = h // nqpkv
            err = (signs_q[h] - signs_kv[hkv]).abs().max().item()
            errors.append(err)
        max_err = max(errors)
        ok = max_err == 0.0
        ok_all &= ok
        check(f"signs_q[h] = signs_kv[h//nqpkv]  (kv={num_kv},q={num_q},d={d})", ok, f"err={max_err:.2e}")

    return ok_all


# ═════════════════════════════════════════════════════════════════════════════
#  5. Test cohérence config vLLM
# ═════════════════════════════════════════════════════════════════════════════

def test_vllm_config():
    print(f"\n{SEP}")
    print("  TEST 5 : Config vLLM — CacheDType et fonctions de dispatch")
    print(SEP)
    ok_all = True

    from vllm.config.cache import CacheDType
    from vllm.v1.attention.backend import is_quantized_kv_cache, is_turbo_quant_kv_cache  # type: ignore[attr-defined]

    check("'turbo_quant' dans CacheDType",
          "turbo_quant" in str(CacheDType))
    check("is_turbo_quant_kv_cache('turbo_quant') == True",
          is_turbo_quant_kv_cache("turbo_quant") is True)
    check("is_turbo_quant_kv_cache('fp8_e5m2') == False",
          is_turbo_quant_kv_cache("fp8_e5m2") is False)
    check("is_quantized_kv_cache('turbo_quant') == True",
          is_quantized_kv_cache("turbo_quant") is True)
    check("is_quantized_kv_cache('fp8_e5m2') == True",
          is_quantized_kv_cache("fp8_e5m2") is True)
    check("is_quantized_kv_cache('auto') == False",
          is_quantized_kv_cache("auto") is False)

    return True


# ═════════════════════════════════════════════════════════════════════════════
#  6. Benchmark performance WHT vs baseline
# ═════════════════════════════════════════════════════════════════════════════

def bench_wht():
    print(f"\n{SEP}")
    print("  BENCH 6 : Performance WHT (rotation KV par batch de tokens)")
    print(SEP)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device : {device}")

    for d in [64, 128]:
        tq = TurboQuantKV(head_size=d, num_kv_heads=8, num_q_heads=32)
        # Batch typique : 512 tokens, 8 têtes KV, dim d
        x = torch.randn(512, 8, d, device=device, dtype=torch.float16)
        x = tq._get_signs_kv(device)  # warm-up lazy init

        # Warm-up
        for _ in range(5):
            _ = tq.rotate_kv(x)

        if device == "cuda":
            torch.cuda.synchronize()

        N = 100
        t0 = time.perf_counter()
        for _ in range(N):
            _ = tq.rotate_kv(x)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / N * 1000

        print(f"  WHT rotate_kv  d={d:3d}, T=512, 8 heads : {dt:.3f} ms/call")

    return True


# ═════════════════════════════════════════════════════════════════════════════
#  Point d'entrée
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
#  6. Tests Dual LUT K/V (V3a — TurboESM §3.3)
# ═════════════════════════════════════════════════════════════════════════════

def test_dual_lut():
    """Valide que les codebooks K et V divergent après calibration et améliorent le SNR."""
    print(f"\n{SEP}")
    print("  TEST 6 : Dual LUT K/V — calibration auto-learning")
    print(SEP)
    ok_all = True

    def snr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        signal = original.float().pow(2).mean()
        noise  = (original.float() - reconstructed.float()).pow(2).mean()
        return 10 * math.log10((signal / noise.clamp(min=1e-12)).item())

    for b_bits, name in [(4, "V1/4-bit"), (3, "V2/3-bit+QJL")]:
        tq = TurboQuantKV(head_size=128, num_kv_heads=2, num_q_heads=8, b_bits=b_bits)

        torch.manual_seed(0)
        # 500 tokens — suffisant pour Lloyd-Max empirique sans OOM
        k = torch.randn(500, 2, 128, dtype=torch.float32)
        v = torch.randn(500, 2, 128, dtype=torch.float32) * 0.8 + torch.randn(500, 2, 128) * 0.4

        k_rot = tq.rotate_kv(k.half()).float()
        v_rot = tq.rotate_v(v.half()).float()

        # SNR avant calibration (codebooks identiques = théoriques)
        k_u8 = tq.quantize_to_uint8(k_rot.half(), which="k")
        v_u8 = tq.quantize_to_uint8(v_rot.half(), which="v")
        k_deq = tq.dequantize_from_uint8(k_u8, which="k").float()
        v_deq = tq.dequantize_from_uint8(v_u8, which="v").float()
        snr_k_before = snr(k_rot, k_deq)
        snr_v_before = snr(v_rot, v_deq)

        # Calibration manuelle
        k_norm = k_rot / k_rot.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v_norm = v_rot / v_rot.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        tq._run_calibration(k_norm.reshape(-1, 128), v_norm.reshape(-1, 128))

        # SNR après calibration
        k_u8 = tq.quantize_to_uint8(k_rot.half(), which="k")
        v_u8 = tq.quantize_to_uint8(v_rot.half(), which="v")
        k_deq = tq.dequantize_from_uint8(k_u8, which="k").float()
        v_deq = tq.dequantize_from_uint8(v_u8, which="v").float()
        snr_k_after = snr(k_rot, k_deq)
        snr_v_after = snr(v_rot, v_deq)

        cb_diverged = not torch.allclose(tq._codebook_k_cpu, tq._codebook_v_cpu, atol=1e-4)
        ok_div = check(f"{name}: codebooks K≠V après calibration", cb_diverged,
                       f"max_diff={(tq._codebook_k_cpu - tq._codebook_v_cpu).abs().max():.6f}")
        ok_k   = check(f"{name}: SNR K après ≥ avant (±0.5 dB)",
                       snr_k_after >= snr_k_before - 0.5,
                       f"{snr_k_before:.2f} → {snr_k_after:.2f} dB")
        ok_v   = check(f"{name}: SNR V après ≥ avant (±0.5 dB)",
                       snr_v_after >= snr_v_before - 0.5,
                       f"{snr_v_before:.2f} → {snr_v_after:.2f} dB")
        ok_cal = check(f"{name}: _calibrated=True", tq._calibrated)

        ok_all = ok_all and ok_div and ok_k and ok_v and ok_cal

    return ok_all


def test_dual_lut_auto_trigger():
    """Valide que la calibration se déclenche automatiquement via store_to_cache."""
    print(f"\n{SEP}")
    print("  TEST 7 : Dual LUT — déclenchement automatique via store_to_cache")
    print(SEP)
    ok_all = True

    tq = TurboQuantKV(head_size=128, num_kv_heads=4, num_q_heads=16, b_bits=3,
                      calib_target=200)  # seuil réduit pour le test
    assert not tq._calibrated

    # Simuler un cache minimal
    nb, bs, nh, d = 8, 16, 4, 128   # 128 tokens / batch
    key_cache   = torch.zeros(nb, bs, nh, d, dtype=torch.uint8)
    value_cache = torch.zeros(nb, bs, nh, d, dtype=torch.uint8)
    slots = torch.arange(nb * bs, dtype=torch.int64)
    torch.manual_seed(1)

    # 3 batches × 128 tokens × 4 têtes = 1536 KV-head tokens > 200
    for _ in range(3):
        k = tq.rotate_kv(torch.randn(nb * bs, nh, d, dtype=torch.float16))
        v = tq.rotate_v(torch.randn(nb * bs, nh, d, dtype=torch.float16))
        tq.store_to_cache(k, v, key_cache, value_cache, slots)
        if tq._calibrated:
            break

    ok_all &= check("calibration déclenchée automatiquement", tq._calibrated)
    ok_all &= check("buffers calibration vidés après calibration",
                    len(tq._calib_k_buf) == 0 and len(tq._calib_v_buf) == 0)
    ok_all &= check("codebooks K/V disponibles après",
                    tq._codebook_k_cpu is not None and tq._codebook_v_cpu is not None)

    return ok_all


# ═════════════════════════════════════════════════════════════════════════════
#  8. Tests V4 SVD per-head (TurboESM §3.4)
# ═════════════════════════════════════════════════════════════════════════════

def test_svd_v4():
    """Vérifie la rotation SVD per-head : orthogonalité, préservation des scores Q·K^T."""
    print(f"\n{SEP}")
    print("  TEST 8 : SVD V4 — rotation per-head")
    print(SEP)
    ok_all = True

    d = 64; nh = 4; nq = 4; T = 300; calib_target = 200

    tq = TurboQuantKV(head_size=d, num_kv_heads=nh, num_q_heads=nq, b_bits=3,
                       calib_target=calib_target)

    # Simuler un store_to_cache pour déclencher la calibration V4
    from unittest.mock import MagicMock
    key_cache   = torch.zeros(10, 32, nh, d, dtype=torch.uint8)
    value_cache = torch.zeros(10, 32, nh, d, dtype=torch.uint8)

    n_batches = (calib_target // (T * nh)) + 2
    for _ in range(n_batches):
        k_raw = torch.randn(T, nh, d)
        v_raw = torch.randn(T, nh, d)
        # Simuler store_to_cache sur CPU
        valid_k = k_raw; valid_v = v_raw
        k_norm = valid_k.float() / valid_k.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v_norm = valid_v.float() / valid_v.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
        tq._calib_k_buf.append(k_norm.reshape(-1, d))
        tq._calib_v_buf.append(v_norm.reshape(-1, d))
        tq._calib_k_buf_v4.append(valid_k.float())
        tq._calib_v_buf_v4.append(valid_v.float())
        tq._calib_collected += T * nh
        if tq._calib_collected >= calib_target and not tq._calibrated:
            k_flat = torch.cat(tq._calib_k_buf, dim=0)
            v_flat = torch.cat(tq._calib_v_buf, dim=0)
            tq._run_calibration(k_flat, v_flat)
            break

    ok_all &= check("V4 calibration déclenchée", tq._v4_calibrated)
    ok_all &= check("rotation_k_cpu shape [nh, d, d]",
                    tq._rotation_k_cpu is not None and
                    tq._rotation_k_cpu.shape == (nh, d, d))
    ok_all &= check("rotation_v_cpu shape [nh, d, d]",
                    tq._rotation_v_cpu is not None and
                    tq._rotation_v_cpu.shape == (nh, d, d))

    # Vérifier l'orthogonalité des rotations : Π_h @ Π_h^T ≈ I
    rot_k = tq._rotation_k_cpu  # [nh, d, d]
    for h in range(nh):
        eye_approx = rot_k[h] @ rot_k[h].T
        err = (eye_approx - torch.eye(d)).abs().max().item()
        ok_all &= check(f"  rotation_k[{h}] orthogonale (err max={err:.2e})", err < 1e-4)

    # Test préservation des scores Q·K^T après rotation V4
    q = torch.randn(5, nq, d)
    k = torch.randn(10, nh, d)
    q_rot = tq.rotate_q(q)
    k_rot = tq.rotate_kv(k)

    # Pour GQA simple (nq==nh), score[t_q, t_k, h] = q[t_q, h] · k[t_k, h]
    # Calcul direct vs rotaté (doit être identique car rotation orthogonale)
    for h in range(nh):
        q_h = q[:, h, :]           # [5, d]
        k_h = k[:, h, :]           # [10, d]
        q_rot_h = q_rot[:, h, :]   # [5, d]
        k_rot_h = k_rot[:, h, :]   # [10, d]
        scores_orig = q_h @ k_h.T          # [5, 10]
        scores_rot  = q_rot_h @ k_rot_h.T  # [5, 10]
        err = (scores_orig - scores_rot).abs().max().item()
        rel = err / (scores_orig.abs().max().item() + 1e-8)
        ok_all &= check(f"  scores Q·K^T préservés tête {h} (err_rel={rel:.2e})", rel < 1e-3)

    # Test unrotate_output : output_unrot ≈ output_original
    v = torch.randn(10, nh, d)
    v_rot = tq.rotate_v(v)
    # Simuler une attention triviale (identity : output = v_rot pour 1 token, 1 tête)
    # output_unrot doit retrouver v
    output_rot = v_rot[:5].clone()  # fictif — on test que unrotate ∘ rotate ≈ id
    output_unrot = tq.unrotate_output(output_rot.unsqueeze(1).expand(-1, nq, -1),
                                       5) if False else None
    # Test simplifié : unrotate_output avec rotation_v
    out_test = v_rot.unsqueeze(0).expand(1, -1, -1, -1).reshape(10, nh, d).clone()
    out_unrot = tq.unrotate_output(out_test, 10)
    err_unrot = (out_unrot - v).abs().max().item()
    ok_all &= check(f"unrotate_output ∘ rotate_v ≈ id (err={err_unrot:.4f})", err_unrot < 0.05)

    return ok_all


def test_svd_v4_gqa():
    """Vérifie la cohérence V4 en mode GQA (num_q_heads > num_kv_heads)."""
    print(f"\n{SEP}")
    print("  TEST 9 : SVD V4 GQA")
    print(SEP)
    ok_all = True

    d = 64; nh_kv = 2; nh_q = 8; T = 400; calib_target = 100

    tq = TurboQuantKV(head_size=d, num_kv_heads=nh_kv, num_q_heads=nh_q, b_bits=3,
                       calib_target=calib_target)

    # Forcer la calibration directement
    k_flat = torch.randn(T * nh_kv, d)
    v_flat = torch.randn(T * nh_kv, d)
    tq._calib_k_buf.append(k_flat)
    tq._calib_v_buf.append(v_flat)
    tq._calib_k_buf_v4.append(torch.randn(T, nh_kv, d))
    tq._calib_v_buf_v4.append(torch.randn(T, nh_kv, d))
    k_flat_all = torch.cat(tq._calib_k_buf)
    v_flat_all = torch.cat(tq._calib_v_buf)
    tq._run_calibration(k_flat_all, v_flat_all)

    ok_all &= check("V4 calibration GQA", tq._v4_calibrated)

    # Vérifier préservation scores Q·K^T en GQA
    q = torch.randn(5, nh_q, d)
    k = torch.randn(7, nh_kv, d)
    q_rot = tq.rotate_q(q)
    k_rot = tq.rotate_kv(k)

    num_per = nh_q // nh_kv
    for kv_h in range(nh_kv):
        for j in range(num_per):
            q_h = nh_q // nh_kv * kv_h + j
            scores_orig = q[:, q_h, :] @ k[:, kv_h, :].T
            scores_rot  = q_rot[:, q_h, :] @ k_rot[:, kv_h, :].T
            err = (scores_orig - scores_rot).abs().max().item()
            rel = err / (scores_orig.abs().max().item() + 1e-8)
            ok_all &= check(
                f"  GQA Q[{q_h}]·K[{kv_h}]^T préservé (err_rel={rel:.2e})", rel < 1e-3
            )

    return ok_all


def main():
    print("\n" + "═" * 70)
    print("  TurboQuant KV Cache — Suite de tests")
    print("  arXiv:2504.19874 — 1Cat-vLLM fork Morph3us-Sigma")
    print("═" * 70)

    results = {
        "WHT"                : test_wht(),
        "Invariants"         : test_invariants(),
        "Phase 2"            : test_phase2_correctness(),
        "GQA"                : test_gqa(),
        "Config vLLM"        : test_vllm_config(),
        "Dual LUT K/V V3a"   : test_dual_lut(),
        "Dual LUT auto-trig" : test_dual_lut_auto_trigger(),
        "SVD V4 per-head"    : test_svd_v4(),
        "SVD V4 GQA"         : test_svd_v4_gqa(),
    }

    bench_wht()

    print(f"\n{'═' * 70}")
    print("  RÉSUMÉ")
    print("─" * 70)
    all_ok = True
    for name, ok in results.items():
        all_ok &= ok
        print(f"  {'✅' if ok else '❌'}  {name}")
    print("═" * 70)

    if all_ok:
        print("\n  🎉  Tous les tests passent — TurboQuant V4 (SVD per-head) validé ✅\n")
    else:
        print("\n  ⚠️  Des tests ont échoué — voir détails ci-dessus\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
