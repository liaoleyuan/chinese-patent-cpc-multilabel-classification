import os
import glob
import json
import ast
import argparse
import random
from collections import Counter
from typing import List, Dict, Tuple

import pandas as pd


def parse_labels_cell(x):
    if pd.isna(x):
        return []
    s = str(x)
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return v
    except Exception:
        pass
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return v
    except Exception:
        pass
    return []


def normalize_label(c: str) -> str:
    return str(c).replace(" ", "").strip()


def filter_g06f_labels(labels: List[str], domain_prefix="G06F") -> List[str]:
    out = []
    seen = set()
    for c in labels:
        if not isinstance(c, str):
            continue
        c = normalize_label(c)
        if c.startswith(domain_prefix) and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def iter_rows_from_csv(path: str, chunksize=200000):
    for chunk in pd.read_csv(path, chunksize=chunksize):
        for _, row in chunk.iterrows():
            pub = str(row.get("publication_number", "")).strip()
            text = str(row.get("text", "") or "").strip()
            labels = parse_labels_cell(row.get("labels", "[]"))
            yield pub, text, labels


def iter_rows_from_shards(shards_glob: str, chunksize=200000):
    files = sorted(glob.glob(shards_glob))
    for fp in files:
        for chunk in pd.read_csv(fp, chunksize=chunksize):
            for _, row in chunk.iterrows():
                pub = str(row.get("publication_number", "")).strip()
                text = str(row.get("text", "") or "").strip()
                labels = parse_labels_cell(row.get("labels", "[]"))
                yield pub, text, labels


def count_labels(rows: List[Tuple[str, str, List[str]]]) -> Counter:
    c = Counter()
    for _, _, labs in rows:
        for l in set(labs):
            c[l] += 1
    return c


def to_df(rows: List[Tuple[str, str, List[str]]], selected_set: set) -> pd.DataFrame:
    data = []
    for pub, text, labs in rows:
        labs_sel = [l for l in labs if l in selected_set]
        if len(labs_sel) == 0:
            continue
        data.append((pub, text, json.dumps(labs_sel, ensure_ascii=False), len(set(labs_sel))))
    return pd.DataFrame(data, columns=["publication_number", "text", "labels", "n_labels"])


def save_ids(path: str, ids: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        for x in ids:
            f.write(str(x) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_csv", required=True, type=str)
    ap.add_argument("--new_shards_glob", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)

    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)

    ap.add_argument("--target_labels", type=int, default=500)
    # 关键参数：按你要比较到的最高N来设，比如100或120
    ap.add_argument("--max_train_per_label", type=int, default=100)
    ap.add_argument("--val_per_label", type=int, default=5)
    ap.add_argument("--test_per_label", type=int, default=5)

    ap.add_argument("--domain_prefix", type=str, default="G06F")
    ap.add_argument("--chunksize", type=int, default=200000)
    ap.add_argument("--max_docs", type=int, default=0, help="debug only; 0 means no limit")
    args = ap.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    os.makedirs(args.out_dir, exist_ok=True)

    # -----------------------------
    # Pass 1: build G06F pool, dedup by publication_number
    # -----------------------------
    pool: Dict[str, Tuple[str, List[str], str]] = {}  # pub -> (text, g06f_labels, source)
    source_counts = Counter()
    seen_rows = 0

    def add_row(pub, text, labels, source_name):
        nonlocal pool
        if not pub:
            return
        if pub in pool:
            return
        if not text:
            return
        g06f = filter_g06f_labels(labels, args.domain_prefix)
        if len(g06f) == 0:
            return
        pool[pub] = (text, g06f, source_name)
        source_counts[source_name] += 1

    # old csv first
    for pub, text, labels in iter_rows_from_csv(args.old_csv, chunksize=args.chunksize):
        add_row(pub, text, labels, "old_csv")
        seen_rows += 1
        if args.max_docs > 0 and len(pool) >= args.max_docs:
            break

    # then new shards
    if args.max_docs == 0:
        for pub, text, labels in iter_rows_from_shards(args.new_shards_glob, chunksize=args.chunksize):
            add_row(pub, text, labels, "new_shards")

    print("[INFO] G06F pool built")
    print("       unique docs:", len(pool))
    print("       from old_csv:", int(source_counts["old_csv"]))
    print("       from new_shards:", int(source_counts["new_shards"]))

    if len(pool) == 0:
        raise RuntimeError("Empty G06F pool. Check input files.")

    # -----------------------------
    # Split by publication_number
    # -----------------------------
    pubs = list(pool.keys())
    rnd = random.Random(args.split_seed)
    rnd.shuffle(pubs)

    n = len(pubs)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    train_ids = pubs[:n_train]
    val_ids = pubs[n_train:n_train + n_val]
    test_ids = pubs[n_train + n_val:]

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    print("[INFO] split sizes:", len(train_ids), len(val_ids), len(test_ids))

    # -----------------------------
    # Build split rows
    # -----------------------------
    train_rows = []
    val_rows = []
    test_rows = []

    for pub in pubs:
        text, labs, _src = pool[pub]
        row = (pub, text, labs)
        if pub in train_set:
            train_rows.append(row)
        elif pub in val_set:
            val_rows.append(row)
        else:
            test_rows.append(row)

    # -----------------------------
    # Count label freqs by split
    # -----------------------------
    freq_train = count_labels(train_rows)
    freq_val = count_labels(val_rows)
    freq_test = count_labels(test_rows)

    # eligible by highest planned N
    eligible = [
        l for l in freq_train.keys()
        if freq_train[l] >= args.max_train_per_label
        and freq_val[l] >= args.val_per_label
        and freq_test[l] >= args.test_per_label
    ]
    # 排序稳定：先按train频次降序，再字典序
    eligible.sort(key=lambda x: (-freq_train[x], x))

    print(f"[INFO] eligible labels (train>={args.max_train_per_label}, val>={args.val_per_label}, test>={args.test_per_label}): {len(eligible)}")

    if len(eligible) < args.target_labels:
        raise RuntimeError(
            f"Not enough eligible labels: have {len(eligible)}, need {args.target_labels}. "
            f"Try lowering --max_train_per_label or adjusting split ratios."
        )

    selected_labels = eligible[:args.target_labels]
    selected_set = set(selected_labels)

    # -----------------------------
    # Export fixed artifacts
    # -----------------------------
    # train_pool: 不按N抽样，只保留 selected labels，用于后续构建 train_80/train_100/train_120
    train_pool_df = to_df(train_rows, selected_set)

    # val/test fixed：后续所有N共用
    val_fixed_df = to_df(val_rows, selected_set)
    test_fixed_df = to_df(test_rows, selected_set)

    # 只保留 n_labels > 0 的文档（to_df 已经做了）
    print("[INFO] train_pool rows:", len(train_pool_df))
    print("[INFO] val_fixed rows:", len(val_fixed_df))
    print("[INFO] test_fixed rows:", len(test_fixed_df))

    out_selected = os.path.join(args.out_dir, "selected_labels.json")
    out_train_ids = os.path.join(args.out_dir, "train_ids.txt")
    out_val_ids = os.path.join(args.out_dir, "val_ids.txt")
    out_test_ids = os.path.join(args.out_dir, "test_ids.txt")
    out_train_pool = os.path.join(args.out_dir, "train_pool.csv")
    out_val_fixed = os.path.join(args.out_dir, "val_fixed.csv")
    out_test_fixed = os.path.join(args.out_dir, "test_fixed.csv")
    out_meta = os.path.join(args.out_dir, "task_meta.json")

    with open(out_selected, "w", encoding="utf-8") as f:
        json.dump(selected_labels, f, ensure_ascii=False, indent=2)

    save_ids(out_train_ids, train_ids)
    save_ids(out_val_ids, val_ids)
    save_ids(out_test_ids, test_ids)

    train_pool_df.to_csv(out_train_pool, index=False, encoding="utf-8-sig")
    val_fixed_df.to_csv(out_val_fixed, index=False, encoding="utf-8-sig")
    test_fixed_df.to_csv(out_test_fixed, index=False, encoding="utf-8-sig")

    # meta
    selected_train_counts = {l: int(freq_train.get(l, 0)) for l in selected_labels}
    meta = {
        "domain_prefix": args.domain_prefix,
        "split_seed": args.split_seed,
        "ratios": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio
        },
        "target_labels": args.target_labels,
        "max_train_per_label_for_eligibility": args.max_train_per_label,
        "val_per_label_for_eligibility": args.val_per_label,
        "test_per_label_for_eligibility": args.test_per_label,
        "pool_stats": {
            "unique_g06f_docs": len(pool),
            "from_old_csv": int(source_counts["old_csv"]),
            "from_new_shards": int(source_counts["new_shards"]),
            "split_sizes": {
                "train_ids": len(train_ids),
                "val_ids": len(val_ids),
                "test_ids": len(test_ids)
            }
        },
        "fixed_rows": {
            "train_pool_rows": int(len(train_pool_df)),
            "val_fixed_rows": int(len(val_fixed_df)),
            "test_fixed_rows": int(len(test_fixed_df))
        },
        "selected_labels_train_count_min": int(min(selected_train_counts.values())),
        "selected_labels_train_count_max": int(max(selected_train_counts.values())),
        "selected_labels_train_counts_top10_low": sorted(selected_train_counts.items(), key=lambda x: x[1])[:10]
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[DONE] wrote:")
    print(" ", out_selected)
    print(" ", out_train_ids, out_val_ids, out_test_ids)
    print(" ", out_train_pool)
    print(" ", out_val_fixed)
    print(" ", out_test_fixed)
    print(" ", out_meta)


if __name__ == "__main__":
    main()
