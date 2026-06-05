import os, glob, json, ast, argparse, random
from collections import defaultdict, Counter

import pandas as pd

def parse_labels_cell(x):
    """
    labels cell in CSV is a JSON string like '["G06F30/20", ...]'
    """
    if pd.isna(x):
        return []
    s = str(x)
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return []

def is_g06f_doc(labels):
    for c in labels:
        if isinstance(c, str) and c.replace(" ", "").startswith("G06F"):
            return True
    return False

def filter_g06f_labels(labels):
    out = []
    seen = set()
    for c in labels:
        if not isinstance(c, str):
            continue
        c = c.replace(" ", "")
        if c.startswith("G06F") and c not in seen:
            seen.add(c)
            out.append(c)
    return out

def iter_rows_from_old_csv(old_csv, chunksize=200000):
    # expects columns: publication_number,text,labels,...
    for chunk in pd.read_csv(old_csv, chunksize=chunksize):
        for _, row in chunk.iterrows():
            pub = str(row.get("publication_number", "")).strip()
            text = str(row.get("text", "") or "").strip()
            labels = parse_labels_cell(row.get("labels", "[]"))
            yield pub, text, labels

def iter_rows_from_shards(shards_glob, chunksize=200000):
    for path in sorted(glob.glob(shards_glob)):
        for chunk in pd.read_csv(path, chunksize=chunksize):
            for _, row in chunk.iterrows():
                pub = str(row.get("publication_number", "")).strip()
                text = str(row.get("text", "") or "").strip()
                labels = parse_labels_cell(row.get("labels", "[]"))
                yield pub, text, labels

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
    ap.add_argument("--train_per_label", type=int, default=120)
    ap.add_argument("--val_per_label", type=int, default=5)
    ap.add_argument("--test_per_label", type=int, default=5)

    ap.add_argument("--max_docs", type=int, default=0, help="0 means no limit; for debugging only")
    args = ap.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6

    os.makedirs(args.out_dir, exist_ok=True)

    # --------- Pass 1: build a doc pool (G06F only), dedup by publication_number ----------
    # Store minimal info in memory: pub -> (text, g06f_labels)
    # For large pools, this can still be big; but for G06F-only it should be manageable.
    pool = {}
    source_counts = Counter()

    def add_row(pub, text, labels, source_name):
        nonlocal pool
        if not pub or pub in pool:
            return
        g06f_labels = filter_g06f_labels(labels)
        if len(g06f_labels) == 0:
            return
        if not text:
            return
        pool[pub] = (text, g06f_labels)
        source_counts[source_name] += 1

    seen = 0
    for pub, text, labels in iter_rows_from_old_csv(args.old_csv):
        add_row(pub, text, labels, "old_csv")
        seen += 1
        if args.max_docs and len(pool) >= args.max_docs:
            break

    if not args.max_docs:
        for pub, text, labels in iter_rows_from_shards(args.new_shards_glob):
            add_row(pub, text, labels, "new_shards")

    print("G06F pool built.")
    print("  unique G06F docs:", len(pool))
    print("  from old_csv:", int(source_counts["old_csv"]))
    print("  from new_shards:", int(source_counts["new_shards"]))

    # --------- Split by publication_number (doc-level split) ----------
    pubs = list(pool.keys())
    rnd = random.Random(args.split_seed)
    rnd.shuffle(pubs)

    n = len(pubs)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    train_pubs = set(pubs[:n_train])
    val_pubs = set(pubs[n_train:n_train+n_val])
    test_pubs = set(pubs[n_train+n_val:])

    print("Split sizes:", len(train_pubs), len(val_pubs), len(test_pubs))

    # --------- Pass 2: count label frequencies in each split ----------
    freq_train = Counter()
    freq_val = Counter()
    freq_test = Counter()

    for pub, (_, labs) in pool.items():
        labs_set = set(labs)
        if pub in train_pubs:
            for l in labs_set: freq_train[l] += 1
        elif pub in val_pubs:
            for l in labs_set: freq_val[l] += 1
        else:
            for l in labs_set: freq_test[l] += 1

    eligible = [
        l for l in freq_train.keys()
        if freq_train[l] >= args.train_per_label and freq_val[l] >= args.val_per_label and freq_test[l] >= args.test_per_label
    ]
    eligible.sort(key=lambda x: freq_train[x], reverse=True)

    print(f"Eligible labels (train>={args.train_per_label},val>={args.val_per_label},test>={args.test_per_label}): {len(eligible)}")
    if len(eligible) < args.target_labels:
        raise RuntimeError(
            f"Not enough eligible labels: have {len(eligible)} need {args.target_labels}. "
            f"Try lowering train_per_label or adjusting split ratios."
        )

    selected_labels = eligible[:args.target_labels]
    label_set = set(selected_labels)

    # --------- Pass 3: build datasets by sampling to satisfy per-label quotas ----------
    # We'll greedily sample documents; a doc can satisfy multiple labels (multi-label overlap).
    need_train = {l: args.train_per_label for l in selected_labels}
    need_val = {l: args.val_per_label for l in selected_labels}
    need_test = {l: args.test_per_label for l in selected_labels}

    def build_split(pubs_set, need, split_name):
        rows = []
        # order docs: more labels first helps satisfy quotas faster
        docs = [(pub, pool[pub][0], [l for l in pool[pub][1] if l in label_set]) for pub in pubs_set]
        docs = [d for d in docs if len(d[2]) > 0]
        docs.sort(key=lambda x: len(set(x[2])), reverse=True)

        for pub, text, labs in docs:
            if all(v <= 0 for v in need.values()):
                break
            # check if this doc helps any still-needed label
            helps = False
            for l in set(labs):
                if need.get(l, 0) > 0:
                    helps = True
                    break
            if not helps:
                continue

            # take doc
            rows.append((pub, text, labs))
            for l in set(labs):
                if need.get(l, 0) > 0:
                    need[l] -= 1

        remaining = sum(1 for v in need.values() if v > 0)
        if remaining > 0:
            # provide debug info: which labels not satisfied
            missing = sorted([(l, need[l]) for l in need if need[l] > 0], key=lambda x: x[1], reverse=True)[:20]
            raise RuntimeError(f"{split_name} build failed: {remaining} labels still missing quota. Top missing: {missing}")
        return rows

    train_rows = build_split(train_pubs, need_train, "train")
    val_rows = build_split(val_pubs, need_val, "val")
    test_rows = build_split(test_pubs, need_test, "test")

    # --------- Write outputs ----------
    def write_csv(path, rows):
        df = pd.DataFrame(rows, columns=["publication_number", "text", "labels"])
        df["labels"] = df["labels"].apply(lambda x: json.dumps(x, ensure_ascii=False))
        df.to_csv(path, index=False)

    tag = f"g06f_{args.target_labels}x{args.train_per_label}"

    out_train = os.path.join(args.out_dir, f"{tag}_train.csv")
    out_val = os.path.join(args.out_dir, f"{tag}_val.csv")
    out_test = os.path.join(args.out_dir, f"{tag}_test.csv")
    out_labels = os.path.join(args.out_dir, f"{tag}_labels.json")

    write_csv(out_train, train_rows)
    write_csv(out_val, val_rows)
    write_csv(out_test, test_rows)

    with open(out_labels, "w", encoding="utf-8") as f:
        json.dump(selected_labels, f, ensure_ascii=False, indent=2)

    out_meta = os.path.join(args.out_dir, f"{tag}_meta.json")
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump({
            "old_csv": args.old_csv,
            "new_shards_glob": args.new_shards_glob,
            "split_seed": args.split_seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "target_labels": args.target_labels,
            "train_per_label": args.train_per_label,
            "val_per_label": args.val_per_label,
            "test_per_label": args.test_per_label,
            "pool_size": len(pool)
        }, f, ensure_ascii=False, indent=2)

    print("DONE. Wrote:")
    print(" ", out_train, "rows=", len(train_rows))
    print(" ", out_val, "rows=", len(val_rows))
    print(" ", out_test, "rows=", len(test_rows))
    print(" ", out_labels, "labels=", len(selected_labels))

if __name__ == "__main__":
    main()
