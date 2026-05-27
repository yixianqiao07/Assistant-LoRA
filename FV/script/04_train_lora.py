# -*- coding: utf-8 -*-
"""
Run one Qwen LoRA training job on an explicit train/dev split.

This script does not create a split and does not augment data. It only consumes
the files passed through --train-path and --dev-path.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FV_ROOT = SCRIPT_DIR.parent
SRC_DIR = FV_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import train_utils


DEFAULT_TRAIN = FV_ROOT / "data" / "train_augmented" / "train_augmented.jsonl"
DEFAULT_DEV = FV_ROOT / "data" / "split" / "dev_original.jsonl"
DEFAULT_OUTPUT = FV_ROOT / "outputs" / "single_lora"


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return (FV_ROOT / path).resolve()


def get_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for training.") from exc
    return torch


def default_device():
    try:
        torch = get_torch()
    except RuntimeError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def validate_data(rows, path, required_fields):
    missing = []
    for index, row in enumerate(rows[:20], 1):
        absent = [field for field in required_fields if field not in row]
        if absent:
            missing.append((index, absent))
    if missing:
        raise ValueError(f"{path} missing required fields: {missing[:5]}")


def build_run_config(args, train_path, dev_path, model_path, output_dir):
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_path": str(train_path),
        "dev_path": str(dev_path),
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "tag": args.tag,
        "device": args.device,
        "multi_gpu": args.multi_gpu,
        "quantize_4bit": args.quantize_4bit,
        "lora": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "max_len": args.max_len,
            "val_every": args.val_every,
            "val_batch_size": args.val_batch_size,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN))
    parser.add_argument("--dev-path", default=str(DEFAULT_DEV))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--tag", default="single_lora")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--quantize-4bit", action="store_true")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--val-every", type=int, default=40)
    parser.add_argument("--val-batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--measure-speed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    torch = get_torch()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_path = resolve_path(args.train_path)
    dev_path = resolve_path(args.dev_path)
    model_path = resolve_path(args.model_path)
    output_dir = resolve_path(args.output_dir) / args.tag

    train_data = train_utils.load_jsonl(train_path)
    dev_data = train_utils.load_jsonl(dev_path)
    required = ["instruction", "input", "output", "answer"]
    validate_data(train_data, train_path, required)
    validate_data(dev_data, dev_path, required)

    run_config = build_run_config(args, train_path, dev_path, model_path, output_dir)

    if args.dry_run:
        print("Dry run OK")
        print(f"Train: {train_path} ({len(train_data)} rows)")
        print(f"Dev: {dev_path} ({len(dev_data)} rows)")
        print(f"Model: {model_path}")
        print(f"Output: {output_dir}")
        print(json.dumps(run_config, ensure_ascii=False, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    train_utils.save_json(run_config, output_dir / "run_config.json")
    logger = train_utils.get_logger(args.tag, output_dir / "train.log")

    logger.info("Run config: %s", json.dumps(run_config, ensure_ascii=False, sort_keys=True))
    logger.info("Loaded train=%s dev=%s", len(train_data), len(dev_data))

    device = torch.device(args.device)
    model, tokenizer = train_utils.load_model_and_tokenizer(
        model_path,
        device,
        quantize_4bit=args.quantize_4bit,
        multi_gpu=args.multi_gpu,
    )
    model = train_utils.apply_lora(
        model,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    )

    trainable, total = model.get_nb_trainable_parameters()
    logger.info("Trainable params: %s / %s (%.4f%%)", trainable, total, 100 * trainable / total)

    started = datetime.now()
    result = train_utils.train_one_run(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=dev_data,
        device=device,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_len=args.max_len,
        save_dir=str(output_dir),
        logger=logger,
        tag=args.tag,
        val_every=args.val_every,
        val_batch_size=args.val_batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    duration_min = round((datetime.now() - started).total_seconds() / 60, 2)
    speed = None
    if args.measure_speed:
        speed = train_utils.measure_inference_speed(
            model,
            tokenizer,
            dev_data,
            device,
            n=min(15, len(dev_data)),
            max_new=args.max_new_tokens,
        )

    result.update(
        {
            "tag": args.tag,
            "duration_min": duration_min,
            "train_size": len(train_data),
            "dev_size": len(dev_data),
            "trainable_params": trainable,
            "total_params": total,
            "speed": speed,
        }
    )

    train_utils.plot_loss_curve(result, output_dir / "loss_curve.png")
    slim_result = {
        key: value
        for key, value in result.items()
        if key not in {"train_losses", "val_losses", "val_accs", "val_steps"}
    }
    train_utils.save_json(result, output_dir / "result_full.json")
    train_utils.save_json(slim_result, output_dir / "result.json")
    logger.info("Finished %s in %.2f min", args.tag, duration_min)
    logger.info("Result: %s", json.dumps(slim_result, ensure_ascii=False, sort_keys=True))

    del model
    train_utils.free_gpu()


if __name__ == "__main__":
    main()
