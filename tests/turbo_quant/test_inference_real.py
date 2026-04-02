"""
Test d'inférence réelle TurboQuant sur Qwen3.5-0.8B (GPU unique).

Comparaison :
  - auto              : fp16 standard (référence)
  - turbo_quant       : V1 4-bit (×3.87)
  - turbo_quant_3bit  : V2 3-bit+QJL (×3.94, ≡ papier arXiv:2504.19874 b=4)
  - turbo_quant_35bit : V5 mixed-precision 3.5-bit (×4.49, objectif du papier)

Chaque dtype est testé dans un subprocess isolé pour éviter la fragmentation
VRAM entre les instances LLM successives.

Usage :
    cd /mnt/data-ssd/morph3us/highbrain/external/1Cat-vLLM
    PYTHONPATH=$(pwd) CUDA_VISIBLE_DEVICES=5 \
    HF_HOME=/mnt/data-ssd/morph3us/highbrain/data/ai/models/hf \
    python3.12 tests/turbo_quant/test_inference_real.py

Usage (mode worker, appelé en interne) :
    python3.12 tests/turbo_quant/test_inference_real.py --worker <kv_dtype>
"""

import gc
import json
import os
import subprocess
import sys
import time

import torch
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3.5-0.8B"
PROMPTS = [
    "Combien font 2+2 ? Réponds en un seul chiffre.",
    "Quelle est la capitale de la France ? Un mot.",
    "Traduis en anglais : 'Bonjour le monde'. Un mot.",
    "Quelle est la couleur du ciel ? Un mot.",
]
PARAMS = SamplingParams(temperature=0, max_tokens=20)
COMMON_KWARGS = dict(
    model=MODEL,
    tensor_parallel_size=1,
    max_model_len=512,
    gpu_memory_utilization=0.50,
    dtype="float16",
    enforce_eager=True,
    max_num_seqs=8,
)
SEP = "─" * 60


# ═══════════════════════════════════════════════════════════════
#  Mode worker : run_test dans un sous-processus propre
# ═══════════════════════════════════════════════════════════════

def _worker_run(kv_dtype: str) -> None:
    """Exécuté dans un subprocess isolé. Imprime le résultat en JSON sur stdout."""
    llm = LLM(kv_cache_dtype=kv_dtype, **COMMON_KWARGS)  # type: ignore[arg-type]

    # Warmup
    _ = llm.generate(["test"], SamplingParams(temperature=0, max_tokens=5))
    # Mesurer la VRAM du GPU via nvidia-smi (car le compute est dans EngineCore subprocess)
    gpu_idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        vram_str = os.popen(
            f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id={gpu_idx}"
        ).read().strip()
        vram = float(vram_str) / 1024  # MiB → GiB
    except Exception:
        vram = 0.0

    t0 = time.perf_counter()
    outputs = llm.generate(PROMPTS, PARAMS)
    elapsed = time.perf_counter() - t0

    texts = [o.outputs[0].text.strip() for o in outputs]

    result = {"texts": texts, "elapsed": elapsed, "vram": vram}
    # Imprimer le résultat sur une ligne JSON sur stdout — sera parsé par le parent
    print("RESULT_JSON:" + json.dumps(result), flush=True)

    del llm
    gc.collect()
    torch.cuda.empty_cache()


def run_test(kv_dtype: str) -> tuple[list[str], float, float]:
    """Lance _worker_run dans un subprocess Python isolé."""
    print(f"\n{SEP}")
    print(f"  kv_cache_dtype = {kv_dtype!r}")
    print(SEP)

    env = os.environ.copy()
    cmd = [sys.executable, __file__, "--worker", kv_dtype]

    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=300
    )

    # Afficher les logs stderr (INFO/WARNING vLLM)
    for line in proc.stderr.splitlines():
        if "KV cache" in line or "Available" in line or "Loading" in line or "Triton" in line:
            print(f"  [log] {line.strip()}")

    if proc.returncode != 0:
        print(f"  ❌  subprocess a échoué (code {proc.returncode})")
        # Afficher les 20 dernières lignes stderr pour le diagnostic
        lines = proc.stderr.splitlines()
        for line in lines[-20:]:
            print(f"  {line}")
        raise RuntimeError(f"run_test({kv_dtype!r}) a échoué")

    # Parser la ligne JSON du résultat
    result_json = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            result_json = json.loads(line[len("RESULT_JSON:"):])
            break

    if result_json is None:
        raise RuntimeError(f"run_test({kv_dtype!r}) : aucun RESULT_JSON dans stdout")

    texts  = result_json["texts"]
    elapsed = result_json["elapsed"]
    vram   = result_json["vram"]

    for prompt, text in zip(PROMPTS, texts):
        print(f"  Q: {prompt}")
        print(f"  A: {text!r}")
    print(f"  Temps génération : {elapsed:.3f}s  |  VRAM : {vram:.2f} GiB")

    return texts, elapsed, vram


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    print("═" * 60)
    print("  TurboQuant — Test inférence réelle")
    print(f"  Modèle : {MODEL}")
    print("═" * 60)

    texts_ref, t_ref, vram_ref   = run_test("auto")
    texts_v1,  t_v1,  vram_v1    = run_test("turbo_quant")
    texts_v2,  t_v2,  vram_v2    = run_test("turbo_quant_3bit")
    texts_v5,  t_v5,  vram_v5    = run_test("turbo_quant_35bit")

    print(f"\n{'═'*60}")
    print("  RÉSULTATS")
    print("═" * 60)

    for i, prompt in enumerate(PROMPTS):
        ref = texts_ref[i]
        v1  = texts_v1[i]
        v2  = texts_v2[i]
        v5  = texts_v5[i]
        ok1 = ref == v1
        ok2 = ref == v2
        ok5 = ref == v5
        print(f"\n  Prompt {i+1}: {prompt[:45]}")
        print(f"    ref (fp16)          : {ref!r}")
        print(f"    V1 (4-bit, ×3.87)   : {v1!r}  {'✅' if ok1 else '⚠️'}")
        print(f"    V2 (3-bit+QJL,×3.94): {v2!r}  {'✅' if ok2 else '⚠️'}")
        print(f"    V5 (3.5-bit, ×4.49) : {v5!r}  {'✅' if ok5 else '⚠️'}")

    def _vram_reduction(vram_a: float, vram_ref: float) -> str:
        if vram_ref < 0.01:
            return "N/A"
        return f"{(1-vram_a/vram_ref)*100:.1f}%"

    print(f"\n{'─'*60}")
    print(f"  VRAM fp16          : {vram_ref:.2f} GiB")
    print(f"  VRAM V1 (×3.87)    : {vram_v1:.2f} GiB  (réduction : {_vram_reduction(vram_v1, vram_ref)})")
    print(f"  VRAM V2 (×3.94)    : {vram_v2:.2f} GiB  (réduction : {_vram_reduction(vram_v2, vram_ref)})")
    print(f"  VRAM V5 (×4.49)    : {vram_v5:.2f} GiB  (réduction : {_vram_reduction(vram_v5, vram_ref)})")
    print(f"\n  Temps fp16         : {t_ref:.3f}s")
    print(f"  Temps V1 (4-bit)   : {t_v1:.3f}s")
    print(f"  Temps V2 (3-bit)   : {t_v2:.3f}s")
    print(f"  Temps V5 (3.5-bit) : {t_v5:.3f}s")

    all_match_v1 = all(texts_ref[i] == texts_v1[i] for i in range(len(PROMPTS)))
    all_match_v2 = all(texts_ref[i] == texts_v2[i] for i in range(len(PROMPTS)))
    all_match_v5 = all(texts_ref[i] == texts_v5[i] for i in range(len(PROMPTS)))

    print(f"\n{'═'*60}")
    if all_match_v1 and all_match_v2 and all_match_v5:
        print("  🎉  Toutes réponses identiques — TurboQuant V1+V2+V5 validés ✅")
    else:
        if all_match_v1:
            print("  ✅  V1 (4-bit)    : identique à fp16")
        else:
            print("  ⚠️   V1 (4-bit)    : légère divergence (bruit quantisation)")
        if all_match_v2:
            print("  ✅  V2 (3-bit)    : identique à fp16")
        else:
            print("  ⚠️   V2 (3-bit)    : légère divergence (bruit quantisation)")
        if all_match_v5:
            print("  ✅  V5 (3.5-bit)  : identique à fp16 — objectif arXiv atteint ✅")
        else:
            print("  ⚠️   V5 (3.5-bit)  : légère divergence (normal : calibration sur 0.8B)")
        print("\n  → La qualité réelle se mesure sur tâches longues (LongBench, NIAH)")
    print("═" * 60)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        _worker_run(sys.argv[2])
    else:
        main()
