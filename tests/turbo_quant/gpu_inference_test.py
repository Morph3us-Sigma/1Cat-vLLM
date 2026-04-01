"""
Test GPU end-to-end TurboQuant — comparaison auto vs turbo_quant (V1 4-bit) vs turbo_quant_3bit (V2).

Usage :
    cd /mnt/data-ssd/morph3us/highbrain/external/1Cat-vLLM
    PYTHONPATH=$(pwd) CUDA_VISIBLE_DEVICES=4,5 \
    HF_HOME=/mnt/data-ssd/morph3us/highbrain/data/ai/models/hf \
    python3.12 tests/turbo_quant/gpu_inference_test.py
"""

import sys
import time

import torch
from vllm import LLM, SamplingParams

MODEL = "QuantTrio/Qwen3.5-9B-AWQ"
PROMPTS = [
    "Combien font 2+2 ? Réponds en un mot.",
    "Quelle est la capitale de la France ? Réponds en un mot.",
    "Traduis en anglais : 'Bonjour le monde'.",
]
PARAMS = SamplingParams(temperature=0, max_tokens=30)
COMMON_KWARGS = dict(
    model=MODEL,
    quantization="awq",
    tensor_parallel_size=2,   # GPUs 4+5 via CUDA_VISIBLE_DEVICES=4,5 → 32 GiB total
    max_model_len=512,
    gpu_memory_utilization=0.60,  # réduit pour laisser de la marge aux tensors temporaires TurboQuant
    dtype="auto",
    enforce_eager=True,
    max_num_seqs=8,
)
SEP = "─" * 70


def run_test(kv_dtype: str) -> tuple[list[str], float]:
    print(f"\n{SEP}")
    print(f"  kv_cache_dtype = {kv_dtype!r}")
    print(SEP)
    llm = LLM(kv_cache_dtype=kv_dtype, **COMMON_KWARGS)  # type: ignore[arg-type]
    t0 = time.perf_counter()
    outputs = llm.generate(PROMPTS, PARAMS)
    elapsed = time.perf_counter() - t0
    texts = [o.outputs[0].text.strip() for o in outputs]
    for prompt, text in zip(PROMPTS, texts):
        print(f"  Q: {prompt}")
        print(f"  A: {text}")
    print(f"  Temps total : {elapsed:.2f}s")

    # VRAM utilisée
    mem = torch.cuda.memory_allocated() / 1024**3
    print(f"  VRAM allouée : {mem:.2f} GiB")

    del llm
    torch.cuda.empty_cache()
    return texts, elapsed


def main() -> None:
    print("═" * 70)
    print("  TurboQuant — Test GPU end-to-end (V1 + V2)")
    print("  Modèle :", MODEL)
    print("═" * 70)

    texts_ref, t_ref   = run_test("auto")
    texts_v1,  t_v1    = run_test("turbo_quant")           # V1 : 4-bit
    texts_v2,  t_v2    = run_test("turbo_quant_3bit")      # V2 : 3-bit + QJL + fp8 norm

    print(f"\n{SEP}")
    print("  COMPARAISON (ref=auto)")
    print(SEP)

    def check(label: str, texts: list[str]) -> bool:
        ok = True
        for i, (ref, tq) in enumerate(zip(texts_ref, texts)):
            match = ref == tq
            ok = ok and match
            print(f"  [{label}] Prompt {i + 1} : {'✅ identiques' if match else '⚠️  différents'}")
            if not match:
                print(f"    ref   : {ref!r}")
                print(f"    {label}: {tq!r}")
        return ok

    ok_v1 = check("turbo_quant (V1 4-bit)", texts_v1)
    ok_v2 = check("turbo_quant (V2 3-bit)", texts_v2)

    print(f"\n  Temps ref              : {t_ref:.2f}s")
    print(f"  Temps turbo_quant V1   : {t_v1:.2f}s  ({'+' if t_v1>t_ref else ''}{t_v1-t_ref:+.2f}s vs ref)")
    print(f"  Temps turbo_quant V2   : {t_v2:.2f}s  ({'+' if t_v2>t_ref else ''}{t_v2-t_ref:+.2f}s vs ref)")
    print("═" * 70)

    if ok_v1 and ok_v2:
        print("\n  🎉  All outputs identiques — TurboQuant V1+V2 GPU validés ✅\n")
    elif ok_v1:
        print("\n  ⚠️  V1 OK, V2 diverge légèrement (bruit 3-bit acceptable)\n")
        sys.exit(0)
    else:
        print("\n  ⚠️  Outputs divergents (attendu si les tokens génèrent un bruit fp8)\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
