# -*- coding: utf-8 -*-
"""Evaluate base and LoRA models on dev/test data."""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FV_ROOT = SCRIPT_DIR.parent
SRC_DIR = FV_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import train_utils


DEFAULT_DATA = FV_ROOT / "data" / "test_processed" / "test_qwen.jsonl"
DEFAULT_OUTPUT = FV_ROOT / "outputs" / "evaluation"


def get_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for model evaluation.") from exc
    return torch


def resolve_path(path):
    if not path:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return (FV_ROOT / path).resolve()


def normalize_answer(answer):
    return "".join(sorted(set((answer or "").strip().upper())))


def extract_answer(text):
    patterns = [
        r"答案[:：]\s*([A-E]+)",
        r"正确答案[:：]?\s*([A-E]+)",
        r"选项[:：]?\s*([A-E]+)",
        r"\b([A-E]{1,5})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "")
        if match:
            return normalize_answer(match.group(1))
    return ""


def build_prompt(sample):
    return (
        f"<|im_start|>system\n{sample['instruction']}<|im_end|>\n"
        f"<|im_start|>user\n{sample['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_jsonl(path):
    return train_utils.load_jsonl(path)


def save_json(obj, path):
    train_utils.save_json(obj, path)


def save_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def accuracy(rows):
    if not rows:
        return 0.0
    return sum(1 for row in rows if row["correct"]) / len(rows)


def grouped_accuracy(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "unknown")].append(row)
    return {
        name: {
            "count": len(items),
            "correct": sum(1 for item in items if item["correct"]),
            "accuracy": round(accuracy(items), 4),
        }
        for name, items in sorted(groups.items())
    }


def validate_samples(samples, path):
    required = ["instruction", "input", "output", "answer"]
    for index, sample in enumerate(samples[:20], 1):
        missing = [field for field in required if field not in sample]
        if missing:
            raise ValueError(f"{path}:{index} missing fields {missing}")


def load_model(base_path, lora_path, device, quantize_4bit=False, multi_gpu=False):
    torch = get_torch()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("transformers is required for evaluation.") from exc

    tokenizer_path = lora_path if lora_path else base_path
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"trust_remote_code": True}
    if quantize_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if device.type == "cuda" else torch.float32

    if multi_gpu:
        kwargs["device_map"] = "auto"
    elif device.type == "cuda":
        kwargs["device_map"] = {"": device}

    model = AutoModelForCausalLM.from_pretrained(str(base_path), **kwargs)
    if device.type != "cuda" and not multi_gpu:
        model.to(device)

    if lora_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("peft is required for LoRA evaluation.") from exc
        model = PeftModel.from_pretrained(model, str(lora_path))

    model.eval()
    return model, tokenizer


def evaluate(model, tokenizer, samples, device, max_new_tokens):
    torch = get_torch()
    rows = []
    started = time.time()

    with torch.no_grad():
        for index, sample in enumerate(samples, 1):
            prompt = build_prompt(sample)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            gold = normalize_answer(sample.get("answer") or extract_answer(sample.get("output", "")))
            pred = extract_answer(generated)
            rows.append(
                {
                    "index": index,
                    "year": sample.get("year", ""),
                    "question_no": sample.get("question_no", ""),
                    "question_type": sample.get("question_type", ""),
                    "subject_area": sample.get("subject_area", ""),
                    "gold_answer": gold,
                    "pred_answer": pred,
                    "correct": pred == gold,
                    "input": sample.get("input", ""),
                    "reference_output": sample.get("output", ""),
                    "prediction_raw": generated,
                }
            )
            if index % 10 == 0 or index == len(samples):
                print(f"  evaluated {index}/{len(samples)}")

    elapsed = round(time.time() - started, 2)
    return rows, elapsed


def build_summary(model_name, model_type, data_path, rows, elapsed_sec):
    return {
        "model_name": model_name,
        "model_type": model_type,
        "data_path": str(data_path),
        "n_samples": len(rows),
        "correct": sum(1 for row in rows if row["correct"]),
        "accuracy": round(accuracy(rows), 4),
        "elapsed_sec": elapsed_sec,
        "by_year": grouped_accuracy(rows, "year"),
        "by_question_type": grouped_accuracy(rows, "question_type"),
        "by_subject_area": grouped_accuracy(rows, "subject_area"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--quantize-4bit", action="store_true")
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_path = resolve_path(args.data)
    output_dir = resolve_path(args.output_dir)
    base_path = resolve_path(args.base_path)
    lora_path = resolve_path(args.lora_path) if args.lora_path else None
    samples = load_jsonl(data_path)
    validate_samples(samples, data_path)

    model_type = "lora" if lora_path else "base"
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model_name.strip())
    pred_path = output_dir / f"{safe_name}_predictions.jsonl"
    summary_path = output_dir / f"{safe_name}_summary.json"

    if args.dry_run:
        print("Dry run OK")
        print(f"Data: {data_path} ({len(samples)} rows)")
        print(f"Base: {base_path}")
        print(f"LoRA: {lora_path}")
        print(f"Model type: {model_type}")
        print(f"Output predictions: {pred_path}")
        print(f"Output summary: {summary_path}")
        return

    torch = get_torch()
    device = torch.device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_name} ({model_type})")
    print(f"Data: {data_path} ({len(samples)} rows)")
    model, tokenizer = load_model(
        base_path=base_path,
        lora_path=lora_path,
        device=device,
        quantize_4bit=args.quantize_4bit,
        multi_gpu=args.multi_gpu,
    )
    rows, elapsed = evaluate(
        model,
        tokenizer,
        samples,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )
    summary = build_summary(args.model_name, model_type, data_path, rows, elapsed)
    save_jsonl(rows, pred_path)
    save_json(summary, summary_path)

    print(
        f"Accuracy: {summary['correct']}/{summary['n_samples']} = "
        f"{summary['accuracy']:.4f}"
    )
    print(f"Wrote {pred_path}")
    print(f"Wrote {summary_path}")

    del model
    train_utils.free_gpu()


if __name__ == "__main__":
    main()
