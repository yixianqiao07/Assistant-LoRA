# -*- coding: utf-8 -*-
"""
Create a leakage-safe train/dev split using subject-area labels.

Default behavior:
  - input:  FV/data/labled.jsonl
  - output: FV/data/split/
  - dev ratio: 12%
  - stratify by: subject_area + question_type

The subject_area label is metadata only. It is preserved in output files for
split auditing and later error analysis, but it is not inserted into the model
prompt.
"""

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FV_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = FV_ROOT / "data" / "labled.jsonl"
DEFAULT_OUTPUT_DIR = FV_ROOT / "data" / "split"

INSTRUCTION = "请完成这道考研政治选择题，给出答案并简要解析。"


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


def save_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean(value):
    return str(value or "").strip()


def normalize_answer(answer):
    return "".join(sorted(set(clean(answer).upper())))


def normalize_for_hash(value):
    return re.sub(r"\s+", "", clean(value)).lower()


def source_id(row, index):
    year = clean(row.get("year")) or "unknown"
    question_no = clean(row.get("question_no")) or str(index + 1)
    return f"{year}-{question_no}"


def content_hash(row):
    parts = [
        row.get("stem", ""),
        row.get("option_A", ""),
        row.get("option_B", ""),
        row.get("option_C", ""),
        row.get("option_D", ""),
        row.get("option_E", ""),
        normalize_answer(row.get("answer")),
    ]
    text = "||".join(normalize_for_hash(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def enrich_rows(rows):
    enriched = []
    for index, row in enumerate(rows):
        item = dict(row)
        item["year"] = clean(item.get("year"))
        item["question_no"] = clean(item.get("question_no"))
        item["question_type"] = clean(item.get("question_type"))
        item["subject_area"] = clean(item.get("subject_area"))
        item["answer"] = normalize_answer(item.get("answer"))
        item["source_id"] = source_id(item, index)
        item["content_hash"] = content_hash(item)
        enriched.append(item)
    return enriched


def parse_stratify_fields(value):
    fields = [item.strip() for item in value.split(",") if item.strip()]
    if not fields:
        raise ValueError("At least one stratify field is required.")
    return fields


def group_key(row, fields):
    return tuple(clean(row.get(field)) or "unknown" for field in fields)


def stratified_split(rows, dev_ratio, seed, stratify_fields):
    rng = random.Random(seed)
    target_dev = max(1, round(len(rows) * dev_ratio))

    groups = defaultdict(list)
    for row in rows:
        groups[group_key(row, stratify_fields)].append(row)

    dev_counts = {}
    remainders = []
    for key, items in groups.items():
        raw_quota = len(items) * dev_ratio
        base = int(raw_quota)
        if len(items) > 1:
            base = min(base, len(items) - 1)
        else:
            base = 0
        dev_counts[key] = base
        remainders.append((raw_quota - int(raw_quota), rng.random(), key))

    remaining = target_dev - sum(dev_counts.values())
    for _, _, key in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        if dev_counts[key] < max(0, len(groups[key]) - 1):
            dev_counts[key] += 1
            remaining -= 1

    dev_ids = set()
    for key, items in groups.items():
        shuffled = items[:]
        rng.shuffle(shuffled)
        for row in shuffled[: dev_counts[key]]:
            dev_ids.add(row["source_id"])

    train_rows = [row for row in rows if row["source_id"] not in dev_ids]
    dev_rows = [row for row in rows if row["source_id"] in dev_ids]
    return train_rows, dev_rows


def options_from_row(row):
    options = []
    for letter in ["A", "B", "C", "D", "E"]:
        value = clean(row.get(f"option_{letter}"))
        if value:
            options.append((letter, value))
    return options


def qtype_label(question_type):
    return "单选题" if question_type == "single" else "多选题"


def to_qwen_sample(row, split):
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
        "instruction": INSTRUCTION,
        "input": input_text,
        "output": f"答案：{answer}\n解析：{explanation}",
        "source_id": row["source_id"],
        "content_hash": row["content_hash"],
        "year": clean(row.get("year")),
        "question_no": clean(row.get("question_no")),
        "question_type": clean(row.get("question_type")),
        "subject_area": clean(row.get("subject_area")),
        "answer": answer,
        "aug_type": "original",
        "split": split,
    }


def summarize(rows, stratify_fields):
    summary = {
        "count": len(rows),
        "by_year": dict(sorted(Counter(row.get("year", "") for row in rows).items())),
        "by_question_type": dict(
            sorted(Counter(row.get("question_type", "") for row in rows).items())
        ),
        "by_subject_area": dict(
            sorted(Counter(row.get("subject_area", "") for row in rows).items())
        ),
    }
    combo_counts = Counter(" | ".join(group_key(row, stratify_fields)) for row in rows)
    summary["by_stratify_group"] = dict(sorted(combo_counts.items()))
    return summary


def assert_no_overlap(train_rows, dev_rows):
    train_source_ids = {row["source_id"] for row in train_rows}
    dev_source_ids = {row["source_id"] for row in dev_rows}
    source_overlap = sorted(train_source_ids & dev_source_ids)

    train_hashes = {row["content_hash"] for row in train_rows}
    dev_hashes = {row["content_hash"] for row in dev_rows}
    hash_overlap = sorted(train_hashes & dev_hashes)

    if source_overlap:
        raise RuntimeError(f"source_id leakage detected: {source_overlap[:10]}")
    if hash_overlap:
        raise RuntimeError(f"content_hash leakage detected: {hash_overlap[:10]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dev-ratio", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratify-fields",
        default="subject_area,question_type",
        help="Comma-separated fields used for stratified sampling.",
    )
    args = parser.parse_args()

    if not 0 < args.dev_ratio < 1:
        raise ValueError("--dev-ratio must be between 0 and 1.")

    stratify_fields = parse_stratify_fields(args.stratify_fields)
    output_dir = Path(args.output_dir)

    rows = enrich_rows(load_jsonl(args.input))
    train_rows, dev_rows = stratified_split(
        rows,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        stratify_fields=stratify_fields,
    )
    assert_no_overlap(train_rows, dev_rows)

    train_qwen = [to_qwen_sample(row, "train") for row in train_rows]
    dev_qwen = [to_qwen_sample(row, "dev") for row in dev_rows]

    save_jsonl(train_rows, output_dir / "raw_train.jsonl")
    save_jsonl(dev_rows, output_dir / "raw_dev.jsonl")
    save_csv(train_rows, output_dir / "raw_train.csv")
    save_csv(dev_rows, output_dir / "raw_dev.csv")
    save_jsonl(train_qwen, output_dir / "train_original.jsonl")
    save_jsonl(dev_qwen, output_dir / "dev_original.jsonl")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(Path(args.input).resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed": args.seed,
        "dev_ratio": args.dev_ratio,
        "stratify_fields": stratify_fields,
        "total": len(rows),
        "train": summarize(train_rows, stratify_fields),
        "dev": summarize(dev_rows, stratify_fields),
        "no_source_id_overlap": True,
        "no_content_hash_overlap": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Input: {Path(args.input).resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Total: {len(rows)}")
    print(f"Train/dev: {len(train_rows)} / {len(dev_rows)}")
    print(f"Dev ratio: {len(dev_rows) / len(rows):.4f}")
    print(f"Stratify fields: {', '.join(stratify_fields)}")
    print(f"Wrote manifest: {output_dir / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
