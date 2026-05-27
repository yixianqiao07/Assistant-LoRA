# -*- coding: utf-8 -*-
"""
Prepare the independent 2024-2026 test set.

This script only standardizes and audits the test data. It does not augment
test samples and does not affect train/dev splitting.
"""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FV_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = FV_ROOT / "data" / "test" / "test_updated.jsonl"
DEFAULT_OUTPUT_DIR = FV_ROOT / "data" / "test_processed"
DEFAULT_TRAIN_RAW = FV_ROOT / "data" / "split" / "raw_train.jsonl"
DEFAULT_DEV_RAW = FV_ROOT / "data" / "split" / "raw_dev.jsonl"

INSTRUCTION = "请完成这道考研政治选择题，给出答案并简要解析。"
LABELS = ["Marxism", "Mao", "History", "Ethics", "Current_Politics"]


KEYWORDS = {
    "Marxism": [
        "马克思主义哲学", "马克思主义基本原理", "哲学", "唯物", "唯心", "辩证",
        "矛盾", "实践", "认识", "真理", "意识", "物质", "历史唯物", "生产力",
        "生产关系", "经济基础", "上层建筑", "商品", "价值", "货币", "资本",
        "剩余价值", "劳动", "价值规律", "垄断", "资本主义", "金融垄断",
    ],
    "Mao": [
        "毛泽东思想", "新民主主义", "社会主义革命", "社会主义改造", "中国特色社会主义",
        "邓小平理论", "三个代表", "科学发展观", "习近平新时代", "马克思主义中国化",
        "共同富裕", "改革开放", "社会主义市场经济", "基本经济制度", "分配制度",
        "党的建设", "中国共产党", "党的领导", "现代化建设", "新发展理念",
        "高质量发展", "乡村振兴", "生态文明", "国家治理", "依法治国",
        "社会主义民主政治", "基层群众自治",
    ],
    "History": [
        "中国近现代史", "近代中国", "鸦片战争", "太平天国", "洋务运动", "戊戌",
        "辛亥革命", "孙中山", "五四运动", "新文化运动", "国民党", "共产党成立",
        "土地革命", "遵义会议", "长征", "抗日战争", "解放战争", "新中国成立",
        "中华人民共和国成立", "十一届三中全会", "历史决议", "甲午", "维新",
        "北伐", "井冈山", "延安", "西柏坡", "民主革命",
    ],
    "Ethics": [
        "思想道德", "道德", "公民道德", "社会主义道德", "法律基础", "宪法",
        "法律", "法治", "权利", "义务", "公民基本权利", "人身权利",
        "人格尊严", "民法", "刑法", "行政法", "婚姻", "继承", "职业道德",
        "家庭美德", "社会公德", "诚信", "理想信念", "人生观", "价值观",
        "爱国主义", "民族精神", "时代精神", "爱情", "核心价值观", "法律义务",
    ],
    "Current_Politics": [
        "时事", "时政", "形势与政策", "国内时事", "国际时事", "重大时事",
        "二十大", "中央经济工作会议", "中央一号文件", "两会", "政府工作报告",
        "国务院新闻办公室", "白皮书", "奥运", "冬奥", "上合组织", "东盟",
        "联合国", "二十国集团", "G20", "APEC", "台湾问题", "南海",
        "一带一路", "全球发展倡议", "全球安全倡议", "北京冬奥会", "习近平主席",
        "国家主席习近平",
    ],
}

HARD_OVERRIDES = {
    "Current_Politics": ["形势与政策", "国内时事", "国际时事", "重大时事", "时政", "时事"],
    "Ethics": ["公民道德", "社会主义道德", "思想道德", "法律义务", "法律权利", "职业道德", "社会公德", "家庭美德", "理想信念", "民族精神", "时代精神"],
    "Mao": ["新民主主义", "毛泽东思想", "邓小平理论", "三个代表", "科学发展观", "习近平新时代", "中国特色社会主义", "马克思主义中国化", "社会主义初级阶段", "社会主义市场经济", "社会主义本质"],
    "History": ["鸦片战争", "洋务运动", "戊戌变法", "辛亥革命", "五四运动", "新文化运动", "抗日战争", "解放战争", "遵义会议", "长征"],
    "Marxism": ["马克思主义哲学", "辩证唯物主义", "历史唯物主义", "唯物史观", "剩余价值", "资本有机构成", "劳动力成为商品"],
}


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
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def drop_output_audit_fields(rows):
    remove = {"source_id", "content_hash", "split"}
    cleaned = []
    for row in rows:
        cleaned.append({key: value for key, value in row.items() if key not in remove})
    return cleaned


def clean(value):
    return str(value or "").strip()


def normalize_answer(answer):
    return "".join(sorted(set(clean(answer).upper())))


def normalize_for_hash(value):
    return re.sub(r"\s+", "", clean(value)).lower()


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


def combined_text(row):
    parts = [
        row.get("stem", ""),
        row.get("option_A", ""),
        row.get("option_B", ""),
        row.get("option_C", ""),
        row.get("option_D", ""),
        row.get("option_E", ""),
        row.get("explanation", ""),
    ]
    return re.sub(r"\s+", "", " ".join(parts))


def question_number_prior(year, question_no):
    try:
        q = int(question_no)
    except (TypeError, ValueError):
        return None

    if 1 <= q <= 4 or 17 <= q <= 20:
        return "Marxism"
    if 5 <= q <= 8 or 21 <= q <= 25:
        return "Mao"
    if 9 <= q <= 12 or 26 <= q <= 29:
        return "History"
    if 13 <= q <= 14 or 30 <= q <= 31:
        return "Ethics"
    if 15 <= q <= 16 or 32 <= q <= 33:
        return "Current_Politics"
    return None


def subject_scores(text):
    scores = {label: 0 for label in LABELS}
    for label, words in KEYWORDS.items():
        for word in words:
            if word and word in text:
                scores[label] += 2 if len(word) >= 4 else 1
    for label, words in HARD_OVERRIDES.items():
        for word in words:
            if word and word in text:
                scores[label] += 20
    return scores


def infer_subject_area(row):
    text = combined_text(row)
    scores = subject_scores(text)
    prior = question_number_prior(row.get("year"), row.get("question_no"))
    if prior:
        scores[prior] += 3
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]


def enrich_rows(rows):
    enriched = []
    for index, row in enumerate(rows):
        item = dict(row)
        item["year"] = clean(item.get("year"))
        item["question_no"] = clean(item.get("question_no"))
        item["question_type"] = clean(item.get("question_type"))
        item["answer"] = normalize_answer(item.get("answer"))
        item["subject_area"] = clean(item.get("subject_area")) or infer_subject_area(item)
        item["source_id"] = f"{item['year']}-{item['question_no']}" if item["year"] and item["question_no"] else f"test-{index + 1}"
        item["content_hash"] = content_hash(item)
        item["split"] = "test"
        enriched.append(item)
    return enriched


def options_from_row(row):
    options = []
    for letter in ["A", "B", "C", "D", "E"]:
        value = clean(row.get(f"option_{letter}"))
        if value:
            options.append((letter, value))
    return options


def qtype_label(question_type):
    return "单选题" if question_type == "single" else "多选题"


def to_qwen_sample(row):
    option_text = "\n".join(f"{letter}. {text}" for letter, text in options_from_row(row))
    input_text = (
        f"题型：{qtype_label(row.get('question_type'))}\n"
        f"题目：{clean(row.get('stem'))}\n"
        f"{option_text}"
    )
    answer = normalize_answer(row.get("answer"))
    return {
        "instruction": INSTRUCTION,
        "input": input_text,
        "output": f"答案：{answer}\n解析：{clean(row.get('explanation'))}",
        "source_id": row["source_id"],
        "content_hash": row["content_hash"],
        "year": row["year"],
        "question_no": row["question_no"],
        "question_type": row["question_type"],
        "subject_area": row["subject_area"],
        "answer": answer,
        "aug_type": "original",
        "split": "test",
    }


def summarize(rows):
    return {
        "count": len(rows),
        "by_year": dict(sorted(Counter(row.get("year", "") for row in rows).items())),
        "by_question_type": dict(sorted(Counter(row.get("question_type", "") for row in rows).items())),
        "by_subject_area": dict(sorted(Counter(row.get("subject_area", "") for row in rows).items())),
    }


def load_hashes(path):
    path = Path(path)
    if not path.exists():
        return set()
    return {row.get("content_hash") for row in load_jsonl(path) if row.get("content_hash")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--train-raw", default=str(DEFAULT_TRAIN_RAW))
    parser.add_argument("--dev-raw", default=str(DEFAULT_DEV_RAW))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows = enrich_rows(load_jsonl(args.input))
    qwen_rows = [to_qwen_sample(row) for row in rows]

    train_hashes = load_hashes(args.train_raw)
    dev_hashes = load_hashes(args.dev_raw)
    test_hashes = {row["content_hash"] for row in rows}
    train_overlap = sorted(test_hashes & train_hashes)
    dev_overlap = sorted(test_hashes & dev_hashes)

    save_jsonl(drop_output_audit_fields(rows), output_dir / "raw_test.jsonl")
    save_csv(drop_output_audit_fields(rows), output_dir / "raw_test.csv")
    save_jsonl(drop_output_audit_fields(qwen_rows), output_dir / "test_qwen.jsonl")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(Path(args.input).resolve()),
        "output_dir": str(output_dir.resolve()),
        "test": summarize(rows),
        "train_hash_overlap_count": len(train_overlap),
        "dev_hash_overlap_count": len(dev_overlap),
        "train_hash_overlap_examples": train_overlap[:10],
        "dev_hash_overlap_examples": dev_overlap[:10],
        "no_train_test_content_hash_overlap": len(train_overlap) == 0,
        "no_dev_test_content_hash_overlap": len(dev_overlap) == 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Input: {Path(args.input).resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Test rows: {len(rows)}")
    print(f"Train/test hash overlap: {len(train_overlap)}")
    print(f"Dev/test hash overlap: {len(dev_overlap)}")
    print(f"Wrote manifest: {output_dir / 'test_manifest.json'}")


if __name__ == "__main__":
    main()
