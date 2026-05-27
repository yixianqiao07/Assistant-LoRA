# -*- coding: utf-8 -*-
"""
Create train-only augmentation variants for controlled experiments.

This script never augments dev/test data. It reads raw_train.jsonl and writes
multiple Qwen SFT-format train files:
  - train_original.jsonl
  - train_instruction_rewrite.jsonl
  - train_shuffle_options.jsonl
  - train_augmented.jsonl
"""

import argparse
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FV_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = FV_ROOT / "data" / "split" / "raw_train.jsonl"
DEFAULT_OUTPUT_DIR = FV_ROOT / "data" / "train_augmented"
DEFAULT_DEV_PATH = FV_ROOT / "data" / "split" / "dev_original.jsonl"

INSTRUCTION = "请完成这道考研政治选择题，给出答案并简要解析。"

INSTRUCTION_VARIANTS = [
    "请分析以下考研政治选择题，选出正确答案并说明理由。",
    "作为考研政治辅导助手，请阅读题目并给出正确选项和简要解释。",
    "请完成下面这道政治选择题，先给出答案，再进行简要解析。",
    "请判断以下考研政治题的正确选项，并概括说明原因。",
    "请根据题干和选项作答，给出答案字母及必要解析。",
]


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


def save_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean(value):
    return str(value or "").strip()


def normalize_answer(answer):
    return "".join(sorted(set(clean(answer).upper())))


def options_from_row(row):
    options = []
    for letter in ["A", "B", "C", "D", "E"]:
        value = clean(row.get(f"option_{letter}"))
        if value:
            options.append((letter, value))
    return options


def qtype_label(question_type):
    return "单选题" if question_type == "single" else "多选题"


def to_qwen_sample(row, instruction=INSTRUCTION, aug_type="original"):
    option_text = "\n".join(
        f"{letter}. {text}" for letter, text in options_from_row(row)
    )
    input_text = (
        f"题型：{qtype_label(row.get('question_type'))}\n"
        f"题目：{clean(row.get('stem'))}\n"
        f"{option_text}"
    )
    answer = normalize_answer(row.get("answer"))
    explanation = clean(row.get("explanation"))
    return {
        "instruction": instruction,
        "input": input_text,
        "output": f"答案：{answer}\n解析：{explanation}",
        "source_id": row["source_id"],
        "content_hash": row["content_hash"],
        "year": clean(row.get("year")),
        "question_no": clean(row.get("question_no")),
        "question_type": clean(row.get("question_type")),
        "subject_area": clean(row.get("subject_area")),
        "answer": answer,
        "aug_type": aug_type,
        "split": "train",
    }


def shuffle_options(row, rng):
    options = options_from_row(row)
    if len(options) < 2:
        return None

    shuffled = options[:]
    rng.shuffle(shuffled)
    if [old for old, _ in shuffled] == [old for old, _ in options]:
        return None

    new_letters = ["A", "B", "C", "D", "E"][: len(shuffled)]
    old_to_new = {
        old_letter: new_letter
        for new_letter, (old_letter, _) in zip(new_letters, shuffled)
    }
    old_answer = normalize_answer(row.get("answer"))
    new_answer = "".join(
        sorted(old_to_new[letter] for letter in old_answer if letter in old_to_new)
    )
    if not new_answer:
        return None

    new_row = dict(row)
    for new_letter, (_, text) in zip(new_letters, shuffled):
        new_row[f"option_{new_letter}"] = text
    for letter in ["A", "B", "C", "D", "E"][len(shuffled) :]:
        new_row[f"option_{letter}"] = ""
    new_row["answer"] = new_answer
    return new_row


def build_train_variants(rows, seed, instruction_rewrites, enable_shuffle):
    rng = random.Random(seed)

    original = [to_qwen_sample(row, aug_type="original") for row in rows]

    instruction_samples = list(original)
    for row in rows:
        variants = INSTRUCTION_VARIANTS[:]
        rng.shuffle(variants)
        for idx, instruction in enumerate(variants[:instruction_rewrites], 1):
            instruction_samples.append(
                to_qwen_sample(
                    row,
                    instruction=instruction,
                    aug_type=f"rewrite_instruction_{idx}",
                )
            )

    shuffle_samples = list(original)
    if enable_shuffle:
        for row in rows:
            shuffled = shuffle_options(row, rng)
            if shuffled:
                shuffle_samples.append(
                    to_qwen_sample(shuffled, aug_type="shuffle_options")
                )

    full_samples = list(original)
    full_samples.extend(sample for sample in instruction_samples if sample["aug_type"] != "original")
    full_samples.extend(sample for sample in shuffle_samples if sample["aug_type"] != "original")

    rng.shuffle(instruction_samples)
    rng.shuffle(shuffle_samples)
    rng.shuffle(full_samples)

    return {
        "train_original": original,
        "train_instruction_rewrite": instruction_samples,
        "train_shuffle_options": shuffle_samples,
        "train_augmented": full_samples,
    }


def summarize(rows):
    return {
        "count": len(rows),
        "by_aug_type": dict(sorted(Counter(row.get("aug_type", "") for row in rows).items())),
        "by_subject_area": dict(
            sorted(Counter(row.get("subject_area", "") for row in rows).items())
        ),
        "by_question_type": dict(
            sorted(Counter(row.get("question_type", "") for row in rows).items())
        ),
    }


def assert_no_dev_overlap(variant_rows, dev_rows):
    train_source_ids = {row["source_id"] for row in variant_rows}
    dev_source_ids = {row["source_id"] for row in dev_rows if row.get("source_id")}
    train_hashes = {row["content_hash"] for row in variant_rows}
    dev_hashes = {row["content_hash"] for row in dev_rows if row.get("content_hash")}

    source_overlap = sorted(train_source_ids & dev_source_ids)
    hash_overlap = sorted(train_hashes & dev_hashes)
    if source_overlap:
        raise RuntimeError(f"Train/dev source_id overlap detected: {source_overlap[:10]}")
    if hash_overlap:
        raise RuntimeError(f"Train/dev content_hash overlap detected: {hash_overlap[:10]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dev-path", default=str(DEFAULT_DEV_PATH))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instruction-rewrites", type=int, default=2)
    parser.add_argument("--no-shuffle-options", action="store_true")
    args = parser.parse_args()

    raw_train = load_jsonl(args.input)
    dev_rows = load_jsonl(args.dev_path) if Path(args.dev_path).exists() else []
    variants = build_train_variants(
        raw_train,
        seed=args.seed,
        instruction_rewrites=args.instruction_rewrites,
        enable_shuffle=not args.no_shuffle_options,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(Path(args.input).resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed": args.seed,
        "instruction_rewrites": args.instruction_rewrites,
        "shuffle_options": not args.no_shuffle_options,
        "dev_path": str(Path(args.dev_path).resolve()),
        "variants": {},
        "no_dev_overlap": True,
    }

    for name, rows in variants.items():
        assert_no_dev_overlap(rows, dev_rows)
        save_jsonl(rows, output_dir / f"{name}.jsonl")
        manifest["variants"][name] = summarize(rows)

    (output_dir / "augmentation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Input: {Path(args.input).resolve()}")
    print(f"Output: {output_dir.resolve()}")
    for name, rows in variants.items():
        print(f"{name}: {len(rows)}")
    print(f"Wrote manifest: {output_dir / 'augmentation_manifest.json'}")


if __name__ == "__main__":
    main()
