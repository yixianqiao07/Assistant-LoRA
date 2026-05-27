# -*- coding: utf-8 -*-
"""Run controlled LoRA ablation experiments."""

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


DEFAULT_CONFIG = FV_ROOT / "config" / "ablation_config.json"


def get_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for ablation training.") from exc
    return torch


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return (FV_ROOT / path).resolve()


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_params(defaults, experiment):
    params = dict(defaults)
    for key, value in experiment.items():
        if key not in {"tag", "group", "description", "train_path"}:
            params[key] = value
    return params


def parse_csv_set(value):
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def selected_experiments(config, only=None, groups=None):
    experiments = []
    for experiment in config["experiments"]:
        if only and experiment["tag"] not in only:
            continue
        if groups and experiment.get("group", "") not in groups:
            continue
        experiments.append(experiment)
    return experiments


def experiment_train_path(config, experiment):
    return experiment.get("train_path") or config["data"]["default_train_path"]


def validate_rows(rows, path):
    required = ["instruction", "input", "output", "answer"]
    for index, row in enumerate(rows[:20], 1):
        missing = [field for field in required if field not in row]
        if missing:
            raise ValueError(f"{path}:{index} missing fields {missing}")


def run_one_experiment(
    experiment,
    params,
    config,
    output_dir,
    dev_data,
    logger,
    measure_speed=False,
):
    torch = get_torch()
    model_cfg = config["model"]
    tag = experiment["tag"]
    train_path = resolve_path(experiment_train_path(config, experiment))
    model_path = resolve_path(model_cfg["path"])
    save_dir = output_dir / tag
    save_dir.mkdir(parents=True, exist_ok=True)

    train_data = train_utils.load_jsonl(train_path)
    validate_rows(train_data, train_path)

    device = torch.device(model_cfg.get("device", "cuda:0"))
    torch.manual_seed(int(params["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(params["seed"]))

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tag": tag,
        "group": experiment.get("group", ""),
        "description": experiment.get("description", ""),
        "train_path": str(train_path),
        "dev_path": str(resolve_path(config["data"]["dev_path"])),
        "model": model_cfg,
        "params": params,
    }
    train_utils.save_json(run_config, save_dir / "run_config.json")

    logger.info("")
    logger.info("=" * 80)
    logger.info("Experiment %s [%s]", tag, experiment.get("group", ""))
    logger.info("Train path: %s (%s rows)", train_path, len(train_data))
    logger.info("Params: %s", json.dumps(params, ensure_ascii=False, sort_keys=True))
    logger.info("=" * 80)

    started = datetime.now()
    model, tokenizer = train_utils.load_model_and_tokenizer(
        model_path,
        device,
        quantize_4bit=bool(model_cfg.get("quantize_4bit", False)),
        multi_gpu=bool(model_cfg.get("multi_gpu", False)),
    )
    model = train_utils.apply_lora(
        model,
        rank=int(params["rank"]),
        alpha=int(params["alpha"]),
        dropout=float(params["dropout"]),
    )
    trainable, total = model.get_nb_trainable_parameters()
    logger.info("Trainable params: %s / %s (%.4f%%)", trainable, total, 100 * trainable / total)

    result = train_utils.train_one_run(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=dev_data,
        device=device,
        lr=float(params["lr"]),
        epochs=int(params["epochs"]),
        batch_size=int(params["batch_size"]),
        grad_accum=int(params["grad_accum"]),
        warmup_ratio=float(params["warmup_ratio"]),
        weight_decay=float(params["weight_decay"]),
        max_len=int(params["max_len"]),
        save_dir=str(save_dir),
        logger=logger,
        tag=tag,
        val_every=int(params["val_every"]),
        val_batch_size=int(params["val_batch_size"]),
        max_new_tokens=int(params["max_new_tokens"]),
    )

    duration_min = round((datetime.now() - started).total_seconds() / 60, 2)
    speed = None
    if measure_speed:
        speed = train_utils.measure_inference_speed(
            model,
            tokenizer,
            dev_data,
            device,
            n=min(15, len(dev_data)),
            max_new=int(params["max_new_tokens"]),
        )

    result.update(
        {
            "tag": tag,
            "group": experiment.get("group", ""),
            "description": experiment.get("description", ""),
            "train_path": str(train_path),
            "train_size": len(train_data),
            "dev_size": len(dev_data),
            "duration_min": duration_min,
            "trainable_params": trainable,
            "total_params": total,
            "speed": speed,
            "params": params,
        }
    )
    train_utils.plot_loss_curve(result, save_dir / "loss_curve.png")
    train_utils.save_json(result, save_dir / "result_full.json")
    slim = {
        key: value
        for key, value in result.items()
        if key not in {"train_losses", "val_losses", "val_accs", "val_steps"}
    }
    train_utils.save_json(slim, save_dir / "result.json")

    del model
    train_utils.free_gpu()
    return slim


def load_existing_result(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_summary(summary, path):
    train_utils.save_json(summary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--only", default="", help="Comma-separated experiment tags")
    parser.add_argument("--groups", default="", help="Comma-separated group names")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--measure-speed", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    config_path = Path(args.config)
    config = load_config(config_path)
    only = parse_csv_set(args.only)
    groups = parse_csv_set(args.groups)
    experiments = selected_experiments(config, only=only, groups=groups)

    output_dir = resolve_path(config["output_dir"])
    dev_path = resolve_path(config["data"]["dev_path"])
    dev_data = train_utils.load_jsonl(dev_path)
    validate_rows(dev_data, dev_path)

    if args.dry_run:
        print(f"Config: {config_path.resolve()}")
        print(f"Output: {output_dir}")
        print(f"Dev: {dev_path} ({len(dev_data)} rows)")
        print(f"Experiments: {len(experiments)}")
        for experiment in experiments:
            params = merge_params(config["defaults"], experiment)
            train_path = resolve_path(experiment_train_path(config, experiment))
            train_data = train_utils.load_jsonl(train_path)
            print(
                f"  {experiment['tag']} "
                f"[{experiment.get('group', '')}] "
                f"train={len(train_data)} path={train_path} "
                f"rank={params['rank']} lr={params['lr']} epochs={params['epochs']}"
            )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = train_utils.get_logger("ablation", output_dir / "ablation.log")
    train_utils.save_json(config, output_dir / "resolved_config.json")

    summary_path = output_dir / "ablation_summary.json"
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path.resolve()),
        "output_dir": str(output_dir),
        "dev_path": str(dev_path),
        "dev_size": len(dev_data),
        "experiments": {},
    }

    for experiment in experiments:
        tag = experiment["tag"]
        result_path = output_dir / tag / "result.json"
        if args.skip_existing and result_path.exists():
            logger.info("Skipping existing experiment: %s", tag)
            result = load_existing_result(result_path)
        else:
            params = merge_params(config["defaults"], experiment)
            result = run_one_experiment(
                experiment=experiment,
                params=params,
                config=config,
                output_dir=output_dir,
                dev_data=dev_data,
                logger=logger,
                measure_speed=args.measure_speed,
            )
        summary["experiments"][tag] = result
        write_summary(summary, summary_path)

    if summary["experiments"]:
        best_tag = max(
            summary["experiments"],
            key=lambda item: summary["experiments"][item]["best_acc"],
        )
        summary["best_by_dev_original"] = {
            "tag": best_tag,
            **summary["experiments"][best_tag],
        }
    write_summary(summary, summary_path)
    logger.info("Ablation complete. Summary: %s", summary_path)


if __name__ == "__main__":
    main()
