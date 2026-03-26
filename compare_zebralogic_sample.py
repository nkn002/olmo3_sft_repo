#!/usr/bin/env python
"""Qualitatively compare two models on one ZebraLogic sample.

This script is designed for the `olmes` environment, where `datasets`,
`transformers`, and optionally `vllm` are available.

Example:
    source /home/jovyan/gpus-4-nodes-volume/vy/bias_in_training_data/.bashrc
    conda activate olmes
    CUDA_VISIBLE_DEVICES=0 python /home/jovyan/gpus-4-nodes-volume/vy/bias_in_training_data/olmo3_sft/compare_zebralogic_sample.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ZEBRA_GRID = """
# Example Puzzle 

There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:
 - Each person has a unique name: `Peter`, `Eric`, `Arnold`.
 - Each person has a unique favorite drink: `tea`, `water`, `milk`

## Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

## Answer to the Example Puzzle

{
    "reasoning": "Given Clue 1, we know Peter is in House 2. According to Clue 2, Arnold is directly left of the one who only drinks water. The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. Thus, Peter drinks water, and Eric lives in House 3. Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.",
    "solution": {
        "House 1": {
            "Name": "Arnold",
            "Drink": "tea"
        },
        "House 2": {
            "Name": "Peter",
            "Drink": "water"
        },
        "House 3": {
            "Name": "Eric",
            "Drink": "milk"
        }
    }
}

# Puzzle to Solve 

{puzzle}


# Instruction

Now please solve the above puzzle. Present your reasoning and solution in the following json format:

{json_template}

"""

EASY_SIZES = {"2*2", "2*3", "2*4", "2*5", "2*6", "3*2", "3*3"}


DEFAULT_LOCAL_MODEL = (
    "/home/jovyan/gpus-4-nodes-volume/vy/bias_in_training_data/olmo3_sft/"
    "repro/olmo3-7b-instruct-sft-repro/checkpoints/"
    "olmo3-7b-instruct-sft-repro/step3394-hf"
)
DEFAULT_OFFICIAL_MODEL = "allenai/Olmo-3-7B-Instruct-SFT"
DEFAULT_TOKENIZER = "allenai/Olmo-3-7B-Instruct-SFT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare one ZebraLogic sample across a local checkpoint and the official HF model."
    )
    parser.add_argument("--sample-index", type=int, default=0, help="0-based sample index in ZebraLogic test split.")
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL, help="Path to the local HF-exported checkpoint.")
    parser.add_argument("--official-model", default=DEFAULT_OFFICIAL_MODEL, help="Official HF model id to compare against.")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="Tokenizer/chat template to use for both models.")
    parser.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help="Inference backend to use for each model.",
    )
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling value.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum new tokens to generate.")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=7000,
        help="vLLM runtime max_model_len cap. Useful when free GPU KV-cache is limited.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Sampling seed for reproducibility.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="vLLM GPU memory utilization target.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the combined comparison result as JSON.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only print the sample prompt and gold answer without loading models.",
    )

    # Internal worker mode. This runs one model per subprocess so model memory is fully released.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--input-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--result-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model-label", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def load_sample(sample_index: int) -> tuple[dict[str, Any], int]:
    from datasets import load_dataset

    split = load_dataset("allenai/ZebraLogicBench-private", name="grid_mode", split="test")

    if sample_index < 0 or sample_index >= len(split):
        raise IndexError(
            f"sample_index={sample_index} is out of range for ZebraLogic test split of size {len(split)}"
        )

    raw_doc = split[sample_index]
    native_index = raw_doc["index"] if isinstance(raw_doc, dict) and "index" in raw_doc else sample_index
    processed = process_doc(raw_doc, native_index)
    return processed, len(split)


def process_doc(doc: dict[str, Any], index: int) -> dict[str, Any]:
    size = doc["size"]
    puzzle = doc["puzzle"]
    solution = doc["solution"]
    prompt_str = ZEBRA_GRID.replace("{puzzle}", puzzle)
    json_template = {"reasoning": "___", "solution": {}}
    num_houses = len(solution["rows"])
    columns = solution["header"]
    for i in range(num_houses):
        json_template["solution"][f"House {i + 1}"] = {
            columns[j]: "___" for j in range(1, len(columns))
        }
    prompt_str = prompt_str.replace("{json_template}", json.dumps(json_template, indent=4))

    solution_table: dict[str, dict[str, str]] = {}
    total_cells = 0
    for i in range(num_houses):
        solution_table[f"House {i + 1}"] = {
            columns[j]: solution["rows"][i][j] for j in range(1, len(columns))
        }
        total_cells += len(columns) - 1

    difficulty = "easy" if size in EASY_SIZES else "hard"
    return {
        "index": index,
        "puzzle": puzzle,
        "size": size,
        "solution_table": solution_table,
        "total_cells": total_cells,
        "query": prompt_str,
        "solution": solution,
        "difficulty": difficulty,
    }


def render_chat_prompt(query: str, tokenizer_name: str) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": query}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return query


def run_worker(args: argparse.Namespace) -> None:
    if not args.input_json or not args.result_json or not args.model or not args.model_label:
        raise ValueError("Worker mode requires --input-json, --result-json, --model, and --model-label.")

    payload = json.loads(Path(args.input_json).read_text())
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.model,
            tokenizer=payload["tokenizer_name"],
            trust_remote_code=True,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype="bfloat16",
            max_model_len=args.max_model_len,
        )
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        request_output = llm.generate(payload["formatted_prompt"], sampling_params, use_tqdm=False)[0]
        text = request_output.outputs[0].text
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(payload["tokenizer_name"], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        inputs = tokenizer(payload["formatted_prompt"], return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        torch.manual_seed(args.seed)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    result = {
        "model_label": args.model_label,
        "model": args.model,
        "backend": args.backend,
        "raw_output": text,
        "parsed_json": extract_last_complete_json(text),
    }
    Path(args.result_json).write_text(json.dumps(result, indent=2))


def extract_last_complete_json(text: str) -> dict[str, Any] | None:
    stack: list[int] = []
    last_json_start = None
    last_json_str = None
    for idx, char in enumerate(text):
        if char == "{":
            stack.append(idx)
            if last_json_start is None:
                last_json_start = idx
        elif char == "}":
            if stack:
                stack.pop()
                if not stack and last_json_start is not None:
                    last_json_str = text[last_json_start : idx + 1]
                    last_json_start = None
    if last_json_str is None:
        return None
    try:
        return json.loads(last_json_str.replace("\n", ""))
    except json.JSONDecodeError:
        return None


def print_report(sample: dict[str, Any], total_samples: int, payload: dict[str, Any], results: list[dict[str, Any]]) -> None:
    print(f"ZebraLogic sample_index={payload['sample_index']} of {total_samples - 1}")
    print(f"native_index={sample['index']} size={sample['size']} difficulty={sample['difficulty']}")
    print()
    print("=== Prompt ===")
    print(sample["query"])
    print()
    print("=== Gold Solution Table ===")
    print(json.dumps(sample["solution_table"], indent=2))
    print()
    for result in results:
        print(f"=== {result['model_label']} Raw Output ===")
        print(result["raw_output"])
        print()
        print(f"=== {result['model_label']} Parsed JSON ===")
        parsed = result.get("parsed_json")
        if parsed is None:
            print("No complete JSON object could be parsed.")
        else:
            print(json.dumps(parsed, indent=2))
        print()


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
        return

    sample, total_samples = load_sample(args.sample_index)
    formatted_prompt = render_chat_prompt(sample["query"], args.tokenizer)

    payload = {
        "sample_index": args.sample_index,
        "tokenizer_name": args.tokenizer,
        "formatted_prompt": formatted_prompt,
    }

    if args.prepare_only:
        print_report(sample, total_samples, payload, results=[])
        return

    models = [
        ("Local step3394-hf", args.local_model),
        ("Official Olmo-3-7B-Instruct-SFT", args.official_model),
    ]

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_json = tmp_path / "input.json"
        input_json.write_text(json.dumps(payload))

        for idx, (label, model) in enumerate(models):
            result_json = tmp_path / f"result_{idx}.json"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--input-json",
                str(input_json),
                "--result-json",
                str(result_json),
                "--model",
                model,
                "--model-label",
                label,
                "--backend",
                args.backend,
                "--tokenizer",
                args.tokenizer,
                "--temperature",
                str(args.temperature),
                "--top-p",
                str(args.top_p),
                "--max-tokens",
                str(args.max_tokens),
                "--seed",
                str(args.seed),
                "--max-model-len",
                str(args.max_model_len),
                "--tensor-parallel-size",
                str(args.tensor_parallel_size),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
            ]
            subprocess.run(cmd, check=True)
            results.append(json.loads(result_json.read_text()))

    print_report(sample, total_samples, payload, results)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "sample": {
                        "sample_index": args.sample_index,
                        "native_index": sample["index"],
                        "size": sample["size"],
                        "difficulty": sample["difficulty"],
                        "query": sample["query"],
                        "solution_table": sample["solution_table"],
                    },
                    "models": results,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
