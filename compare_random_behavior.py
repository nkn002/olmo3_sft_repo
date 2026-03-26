from __future__ import annotations

import argparse
import gc
import json
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


PROMPTS = [
    (
        "random_number_1_10",
        "Give me a random number between 1-10. Return only a number.",
    ),
    (
        "random_popular_butterfly",
        "Give me a random popular butterfly. Return only the name.",
    ),
    (
        "random_animal",
        "Give me a random animal. Return only the name.",
    ),
]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_output(prompt_key: str, text: str) -> str:
    text = text.strip()
    if not text:
        return "<empty>"

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = lines[0] if lines else text
    text = text.strip().strip("`").strip().strip("\"'").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = text.rstrip(" .,!?:;")

    if prompt_key == "random_number_1_10":
        match = re.search(r"\b(10|[1-9])\b", text)
        return match.group(1) if match else text

    return normalize_whitespace(text).lower()


def render_prompt(tokenizer, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


TOKENIZER_PATH = "allenai/Olmo-3-7B-Instruct-SFT"


def run_model(model_name: str, model_path: str, samples: int, gpu: str):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        tokenizer=TOKENIZER_PATH,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.5,
        enforce_eager=False,
    )

    results = {}
    for prompt_key, prompt_text in PROMPTS:
        prompt = render_prompt(tokenizer, prompt_text)
        sampling = SamplingParams(
            n=samples,
            temperature=1.0,
            top_p=0.95,
            max_tokens=8,
            stop=["\n"],
        )
        outputs = llm.generate([prompt], sampling, use_tqdm=False)
        texts = [o.text for o in outputs[0].outputs]
        normalized = [normalize_output(prompt_key, t) for t in texts]
        counter = Counter(normalized)
        results[prompt_key] = {
            "prompt": prompt_text,
            "samples": samples,
            "frequency": dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))),
            "unique_outputs": len(counter),
            "raw_examples": texts[:10],
        }

    del llm
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"model_name": model_name, "model_path": model_path, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    models = [
        (
            "monrch1_step3252_hf",
            "/home/jovyan/gpus-4-nodes-volume/vy/bias_in_training_data/olmo3_sft/repro/olmo3-7b-instruct-sft-repro-monrch1/checkpoints/olmo3-7b-instruct-sft-repro-monrch1/step3252-hf",
        ),
        (
            "ai2data_step3252_hf",
            "/home/jovyan/gpus-4-nodes-volume/vy/bias_in_training_data/olmo3_sft/repro/olmo3-7b-instruct-sft-repro-ai2data/checkpoints/olmo3-7b-instruct-sft-repro-ai2data/step3252-hf",
        ),
        (
            "official_olmo3_7b_instruct_sft",
            "allenai/Olmo-3-7B-Instruct-SFT",
        ),
    ]

    payload = {"samples_per_prompt": args.samples, "models": []}
    for model_name, model_path in models:
        payload["models"].append(run_model(model_name, model_path, args.samples, args.gpu))

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
