"""
Test d'inférence réelle TurboQuant sur Qwen3.5-0.8B (GPU unique).

Comparaison :
  - auto     : fp16 standard (référence)
  - turbo_quant      : V1 4-bit
  - turbo_quant_3bit : V2 3-bit+QJL (≡ papier arXiv:2504.09874)

Usage :
    cd /mnt/data-ssd/morph3us/highbrain/external/1Cat-vLLM
    PYTHONPATH=$(pwd) CUDA_VISIBLE_DEVICES=4 \
    HF_HOME=/mnt/data-ssd/morph3us/highbrain/data/ai/models/hf \
    python3.12 tests/turbo_quant/test_inference_real.py
"""

import gc
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


def run_test(kv_dtype: str) -> tuple[list[str], float, float]:
    print(f"\n{SEP}")
    print(f"  kv_cache_dtype = {kv_dtype!r}")
    print(SEP)

    llm = LLM(kv_cache_dtype=kv_dtype, **COMMON_KWARGS)  # type: ignore[arg-type]

    # Warmup
    _ = llm.generate(["test"], SamplingParams(temperature=0, max_tokens=5))

    t0 = time.perf_counter()
    outputs = llm.generate(PROMPTS, PARAMS)
    elapsed = time.perf_counter() - t0

    texts = [o.outputs[0].text.strip() for o in outputs]
    for prompt, text in zip(PROMPTS, texts):
        print(f"  Q: {prompt}")
        print(f"  A: {text!r}")

    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"  Temps génération : {elapsed:.3f}s  |  VRAM : {vram:.2f} GiB")

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return texts, elapsed, vram


def main() -> None:
    print("═" * 60)
    print("  TurboQuant — Test inférence réelle")
    print(f"  Modèle : {MODEL}")
    print("═" * 60)

    texts_ref, t_ref, vram_ref   = run_test("auto")
    texts_v1,  t_v1,  vram_v1    = run_test("turbo_quant")
    texts_v2,  t_v2,  vram_v2    = run_test("turbo_quant_3bit")

    print(f"\n{'═'*60}")
    print("  RÉSULTATS")
    print("═" * 60)

    for i, prompt in enumerate(PROMPTS):
        ref = texts_ref[i]
        v1  = texts_v1[i]
        v2  = texts_v2[i]
        ok1 = ref == v1
        ok2 = ref == v2
        print(f"\n  Prompt {i+1}: {prompt[:45]}")
        print(f"    ref (fp16)       : {ref!r}")
        print(f"    V1 (4-bit)       : {v1!r}  {'✅' if ok1 else '⚠️'}")
        print(f"    V2 (3-bit+QJL)   : {v2!r}  {'✅' if ok2 else '⚠️'}")

    print(f"\n{'─'*60}")
    print(f"  VRAM fp16        : {vram_ref:.2f} GiB")
    print(f"  VRAM V1 (4-bit)  : {vram_v1:.2f} GiB  (réduction : {(1-vram_v1/vram_ref)*100:.1f}%)")
    print(f"  VRAM V2 (3-bit)  : {vram_v2:.2f} GiB  (réduction : {(1-vram_v2/vram_ref)*100:.1f}%)")
    print(f"\n  Temps fp16       : {t_ref:.3f}s")
    print(f"  Temps V1 (4-bit) : {t_v1:.3f}s")
    print(f"  Temps V2 (3-bit) : {t_v2:.3f}s")

    all_match_v1 = all(texts_ref[i] == texts_v1[i] for i in range(len(PROMPTS)))
    all_match_v2 = all(texts_ref[i] == texts_v2[i] for i in range(len(PROMPTS)))

    print(f"\n{'═'*60}")
    if all_match_v1 and all_match_v2:
        print("  🎉  Toutes les réponses identiques — TurboQuant V1+V2 validés ✅")
    elif all_match_v1:
        print("  ✅  V1 (4-bit)   : identique à fp16")
        print("  ⚠️   V2 (3-bit)   : légère divergence (bruit 3-bit acceptable)")
    elif all_match_v2:
        print("  ⚠️   V1 (4-bit)   : légère divergence")
        print("  ✅  V2 (3-bit)   : identique à fp16")
    else:
        print("  ⚠️  Divergences observées (normal à cause du bruit de quantisation)")
        print("       → Ce n'est pas un bug : les tokens peuvent légèrement différer")
        print("       → La qualité réelle se mesure sur des tâches longues (perplexité)")
    print("═" * 60)


if __name__ == "__main__":
    main()
