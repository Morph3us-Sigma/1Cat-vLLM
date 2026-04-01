# TurboQuant KV Cache — Intégration 1Cat-vLLM

> Implémentation de **TurboQuant** (Zandieh et al., Google Research, arXiv:2504.19874)  
> dans le fork [1Cat-vLLM](https://github.com/Morph3us-Sigma/1Cat-vLLM) — backend Triton V100

---

## Principe

TurboQuant est un algorithme de **quantification vectorielle online** du KV cache,
data-oblivious, sans calibration, applicable en temps réel.

**Algorithme :**
```
R = H · D
```
- `H` : Transformation de Walsh-Hadamard normalisée (orthogonale, $O(d \log d)$)
- `D = diag(s₁,...,sₐ)` avec sᵢ ∈ {±1} tirés aléatoirement (seed fixe par couche)

**Propriété clé** : après rotation par R, les coordonnées suivent ~Beta(d/2, d/2).
Cette distribution concentrée permet une quantification scalaire quasi-optimale coordonnée par coordonnée.

**Résultats publiés :**
| Bits/canal | Qualité vs FP16 | Compression KV |
|---|---|---|
| FP16 (16 bits) | référence | ×1 |
| fp8_e5m2 (8 bits) | ≈ identique | ×2 |
| **3.5 bits** | **neutralité absolue** | **×4.5** |
| 2.5 bits | dégradation marginale | ×6.4 |

---

## Architecture de l'intégration

### Fichiers ajoutés / modifiés

| Fichier | Rôle |
|---|---|
| `vllm/model_executor/layers/quantization/turbo_quant_kv.py` | Module TurboQuant pur (WHT, rotations, un-rotation) |
| `vllm/config/cache.py` | Ajout dtype `"turbo_quant"` dans `CacheDType` |
| `vllm/v1/attention/backend.py` | `is_turbo_quant_kv_cache()` + extension `is_quantized_kv_cache()` |
| `vllm/v1/attention/backends/triton_attn.py` | Injection Phase 1+2 dans le backend Triton V100 |

### Flux de calcul (Phase 2 complète)

```
Prefill / Decode
       │
   K, Q, V  (FP16/BF16)
       │
┌──────▼──────────────────────────────────────────────┐
│  TurboQuantKV.rotate_kv(K)  →  K_rot = R·K          │  do_kv_cache_update()
│  TurboQuantKV.rotate_v(V)   →  V_rot = R·V          │
│  stocker K_rot, V_rot en fp8_e5m2 dans le KV cache   │
└──────────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────┐
│  TurboQuantKV.rotate_q(Q)   →  Q_rot = R·Q          │  forward()
│  Q_rot · K_rot^T = R·Q · (R·K)^T = Q·K^T  ✅        │
│  unified_attention(Q_rot, K_rot_cache, V_rot_cache)  │
│  →  output_rot[t,h] = Σᵢ attn[t,i,h] · V_rot[i,hₖᵥ]│
│               = R_{hₖᵥ} · output[t,h]               │
└──────────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────┐
│  TurboQuantKV.unrotate_output(output_rot)            │  forward()
│  = R⁻¹ · output_rot = output original ✅             │
└──────────────────────────────────────────────────────┘
```

**Invariants vérifiés** (tests unitaires inline) :
- H·H = I (WHT est son propre inverse normalisé)
- Q_rot · K_rot^T = Q · K^T (scores d'attention inchangés)
- V_rot + un-rotation output = résultat identique à FP16 (err = 0.000)

---

## Utilisation

### Activer dans un modèle vLLM

Ajouter dans `special_params` de l'entrée modèle (`config/models.py`) :

```python
"special_params": {
    "reasoning_parser": "qwen3",
    # Phase 0 : fp8 KV sans rotation (÷2 VRAM, déjà supporté nativement)
    "kv_cache_dtype": "fp8_e5m2",
    # Phase 1+2 complète (rebuild 1Cat-vLLM requis) :
    # "kv_cache_dtype": "turbo_quant",
}
```

### Activer via CLI vLLM

```bash
python -m vllm.entrypoints.openai.api_server \
    --model QuantTrio/Qwen3.5-27B-AWQ \
    --kv-cache-dtype turbo_quant \
    --tensor-parallel-size 2
```

### Contraintes

- `head_size` doit être une puissance de 2 (64, 128, 256) — c'est le cas de tous les Qwen3.5
- `kv_cache_dtype = "turbo_quant"` utilise `fp8_e5m2` comme format de stockage interne
- Le backend Flash V100 (`_supports_flash_v100_path`) désactive automatiquement fp8 → fallback Triton avec TurboQuant actif ✅

---

## API du module `TurboQuantKV`

```python
from vllm.model_executor.layers.quantization.turbo_quant_kv import TurboQuantKV

tq = TurboQuantKV(
    head_size=128,       # doit être puissance de 2
    num_kv_heads=8,
    num_q_heads=32,      # GQA supporté
    seed=42,             # reproductibilité (même seed = même rotation)
)

# Encoder (do_kv_cache_update)
K_rot = tq.rotate_kv(K)   # [T, num_kv_heads, head_size]
V_rot = tq.rotate_v(V)    # [T, num_kv_heads, head_size]  (même op que rotate_kv)

# Decoder (forward)
Q_rot = tq.rotate_q(Q)    # [T, num_q_heads, head_size]

# Post-attention (forward)
tq.unrotate_output(output, num_actual_tokens)  # in-place, [T_padded, num_q_heads*head_size]
```

---

## Phases d'implémentation

| Phase | Description | État |
|---|---|---|
| **0** | `fp8_e5m2` KV natif vLLM — ÷2 VRAM, 0 code custom | ✅ actif (`models.py`) |
| **1** | Rotation Q+K uniquement — prépare le terrain pour Phase 2 | ✅ implémenté |
| **2** | Rotation Q+K+V + un-rotation output — résultat identique FP16 | ✅ implémenté |
| **3** (future) | Quantification 3.5-bit custom (int4 packed with optimal scalar quantizer) | 🔄 à implémenter |

La Phase 3 nécessitera un kernel Triton custom pour le dequantize lors du compute d'attention.

---

## Références

- **TurboQuant** : Zandieh, Daliri, Hadian, Mirrokni (Google Research)  
  *"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"*  
  arXiv:2504.19874, 28 avril 2025
- **QJL** : méthode complémentaire pour inner product non-biaisé sur le résidu
- **Walsh-Hadamard Transform** : O(d log d), pas de paramètres appris, data-oblivious

---

*Intégration HighBrain / 1Cat-vLLM — Morph3us-Sigma*
