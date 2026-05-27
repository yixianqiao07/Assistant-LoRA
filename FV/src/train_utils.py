# -*- coding: utf-8 -*-
"""Training utilities for Qwen LoRA SFT."""

import gc
import json
import logging
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset


def get_logger(name, log_file=None):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_prompt(sample):
    return (
        f"<|im_start|>system\n{sample['instruction']}<|im_end|>\n"
        f"<|im_start|>user\n{sample['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_full(sample):
    prompt = build_prompt(sample)
    return prompt, prompt + sample["output"] + "<|im_end|>"


class PoliticsDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt, full = build_full(sample)
        full_enc = self.tokenizer(
            full,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        prompt_enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = full_enc["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[: prompt_enc["input_ids"].shape[1]] = -100
        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch, pad_id):
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids, labels = [], []
    for item in batch:
        pad_len = max_len - item["input_ids"].shape[0]
        input_ids.append(
            torch.cat(
                [
                    item["input_ids"],
                    torch.full((pad_len,), pad_id, dtype=torch.long),
                ]
            )
        )
        labels.append(
            torch.cat(
                [
                    item["labels"],
                    torch.full((pad_len,), -100, dtype=torch.long),
                ]
            )
        )
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": (input_ids != pad_id).long(),
    }


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


def extract_gt(output_text):
    return extract_answer(output_text)


@torch.no_grad()
def compute_val_loss(model, val_ds, pad_id, device, batch_size=2):
    model.eval()
    data_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, pad_id),
    )
    total_loss, count = 0.0, 0
    for batch in data_loader:
        output = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        total_loss += output.loss.item()
        count += 1
    return total_loss / max(count, 1)


@torch.no_grad()
def compute_accuracy(model, tokenizer, val_samples, device, max_new=128):
    model.eval()
    correct, total = 0, 0
    for sample in val_samples:
        enc = tokenizer(build_prompt(sample), return_tensors="pt").to(device)
        output_ids = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(
            output_ids[0][enc["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        gold = normalize_answer(sample.get("answer") or extract_gt(sample.get("output", "")))
        pred = extract_answer(generated)
        correct += int(pred == gold)
        total += 1
    return correct / total if total else 0.0


def load_model_and_tokenizer(model_path, device, quantize_4bit=False, multi_gpu=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
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

    model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
    if device.type != "cuda" and not multi_gpu:
        model.to(device)
    return model, tokenizer


def apply_lora(model, rank=16, alpha=32, dropout=0.05):
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    return get_peft_model(model, config)


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_one_run(
    model,
    tokenizer,
    train_data,
    val_data,
    device,
    lr=2e-4,
    epochs=3,
    batch_size=4,
    grad_accum=4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_len=512,
    save_dir=None,
    logger=None,
    tag="",
    val_every=50,
    val_batch_size=2,
    max_new_tokens=128,
):
    pad_id = tokenizer.pad_token_id
    train_ds = PoliticsDataset(train_data, tokenizer, max_len)
    val_ds = PoliticsDataset(val_data, tokenizer, max_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_id),
        num_workers=0,
    )

    total_steps = max(1, len(train_loader) * epochs // grad_accum)
    warmup_steps = int(total_steps * warmup_ratio)
    optimizer = AdamW(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.1)

    if logger:
        logger.info(
            "[%s] train=%s val=%s steps=%s warmup=%s",
            tag,
            len(train_data),
            len(val_data),
            total_steps,
            warmup_steps,
        )

    train_losses, val_losses, val_accs, val_steps = [], [], [], []
    global_step = 0
    best_acc = -1.0
    best_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            (output.loss / grad_accum).backward()
            epoch_loss += output.loss.item()

            if global_step < warmup_steps:
                for group in optimizer.param_groups:
                    group["lr"] = lr * (global_step + 1) / max(warmup_steps, 1)

            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                current_loss = epoch_loss / step
                train_losses.append(current_loss)

                if logger and global_step % 10 == 0:
                    logger.info(
                        "[%s] epoch=%s step=%s/%s loss=%.4f lr=%.2e",
                        tag,
                        epoch,
                        global_step,
                        total_steps,
                        current_loss,
                        optimizer.param_groups[0]["lr"],
                    )

                if global_step % val_every == 0:
                    val_loss = compute_val_loss(
                        model,
                        val_ds,
                        pad_id,
                        device,
                        batch_size=val_batch_size,
                    )
                    val_acc = compute_accuracy(
                        model,
                        tokenizer,
                        val_data,
                        device,
                        max_new=max_new_tokens,
                    )
                    val_losses.append(val_loss)
                    val_accs.append(val_acc)
                    val_steps.append(global_step)
                    if logger:
                        logger.info(
                            "[%s] val step=%s loss=%.4f acc=%.4f",
                            tag,
                            global_step,
                            val_loss,
                            val_acc,
                        )
                    if val_acc >= best_acc:
                        best_acc = val_acc
                        best_step = global_step
                        if save_dir:
                            best_dir = Path(save_dir) / "best_model"
                            model.save_pretrained(best_dir)
                            tokenizer.save_pretrained(best_dir)
                    model.train()

        if logger:
            logger.info(
                "[%s] epoch=%s avg_loss=%.4f",
                tag,
                epoch,
                epoch_loss / max(len(train_loader), 1),
            )

    final_acc = compute_accuracy(
        model,
        tokenizer,
        val_data,
        device,
        max_new=max_new_tokens,
    )
    if best_acc < 0:
        best_acc = final_acc
        if save_dir:
            best_dir = Path(save_dir) / "best_model"
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

    if logger:
        logger.info("[%s] final_acc=%.4f best_acc=%.4f", tag, final_acc, best_acc)

    return {
        "tag": tag,
        "best_acc": best_acc,
        "best_step": best_step,
        "final_acc": final_acc,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "val_steps": val_steps,
    }


@torch.no_grad()
def measure_inference_speed(model, tokenizer, samples, device, n=20, max_new=128):
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times, token_counts = [], []
    for sample in samples[: min(n, len(samples))]:
        enc = tokenizer(build_prompt(sample), return_tensors="pt").to(device)
        start = time.time()
        output_ids = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.time() - start)
        token_counts.append(output_ids.shape[1] - enc["input_ids"].shape[1])

    avg_time = float(np.mean(times)) if times else 1.0
    avg_tokens = float(np.mean(token_counts)) if token_counts else 0.0
    peak_mb = 0.0
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    return {
        "avg_latency_sec": round(avg_time, 3),
        "tokens_per_sec": round(avg_tokens / avg_time if avg_time > 0 else 0.0, 1),
        "peak_memory_mb": round(peak_mb, 1),
        "peak_memory_gb": round(peak_mb / 1024, 2),
    }


def plot_loss_curve(result, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    steps = list(range(1, len(result["train_losses"]) + 1))
    ax1.plot(steps, result["train_losses"], label="Train Loss", color="#2563EB", lw=1.5)
    if result["val_losses"]:
        ax1.plot(
            result["val_steps"],
            result["val_losses"],
            label="Val Loss",
            color="#DC2626",
            lw=1.5,
            marker="o",
            ms=4,
        )
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss Curve [{result['tag']}]")
    ax1.legend()
    ax1.grid(alpha=0.3)

    if result["val_accs"]:
        ax2.plot(
            result["val_steps"],
            result["val_accs"],
            color="#16A34A",
            lw=1.5,
            marker="o",
            ms=4,
        )
        ax2.axhline(
            y=result["best_acc"],
            color="#DC2626",
            ls="--",
            lw=1,
            label=f"Best={result['best_acc']:.3f}",
        )
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Accuracy")
        ax2.set_title(f"Val Accuracy [{result['tag']}]")
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
