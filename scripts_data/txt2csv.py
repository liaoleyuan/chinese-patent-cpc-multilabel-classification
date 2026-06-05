import ast
import json
import csv
from pathlib import Path
from typing import Optional
from pathlib import Path

def read_one_record_from_txt(txt_path: str):
    """Each txt contains exactly one line (one patent record)."""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            if not line:
                return None
            return ast.literal_eval(line)
    except Exception:
        return None

def load_progress(progress_path: str):
    p = Path(progress_path)
    if not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return set()

def save_progress(progress_path: str, done_files_set: set):
    Path(progress_path).write_text(
        json.dumps(sorted(done_files_set), ensure_ascii=False),
        encoding="utf-8"
    )

def build_multilabel_csv_from_txt_dir(
    txt_dir: str,
    out_csv_path: str,
    progress_path: str = "progress_done_files.json",
    max_docs: Optional[int] = None,
    max_files: Optional[int] = None,
    max_abstract_chars: int = 1200,
    log_every: int = 2000,
    save_progress_every: int = 2000,
):
    txt_paths = sorted([str(p) for p in Path(txt_dir).glob("*.txt")])
    if max_files is not None:
        txt_paths = txt_paths[:max_files]

    done_files = load_progress(progress_path)

    # 如果是断点续跑，文件存在则追加写入；否则新建并写header
    out_exists = Path(out_csv_path).exists()
    mode = "a" if out_exists else "w"

    n_written = 0

    fieldnames = ["publication_number", "text", "labels", "n_labels"]
    with open(out_csv_path, mode, encoding="utf-8-sig", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=fieldnames)
        if not out_exists:
            writer.writeheader()

        for idx, fp in enumerate(txt_paths, 1):
            if fp in done_files:
                continue

            if max_docs is not None and n_written >= max_docs:
                print(f"Reached max_docs={max_docs}, stop.")
                break

            obj = read_one_record_from_txt(fp)
            if obj is None:
                done_files.add(fp)
                continue

            pub = str(obj.get("publication_number", "")).strip()
            title = str(obj.get("title", "")).strip()
            abstract = str(obj.get("abstract", "")).strip()

            if max_abstract_chars is not None and len(abstract) > max_abstract_chars:
                abstract = abstract[:max_abstract_chars]

            kws = obj.get("prior_art_keywords", [])
            if isinstance(kws, list):
                keywords = " ".join([str(x).strip() for x in kws if str(x).strip()])
            elif kws is None:
                keywords = ""
            else:
                keywords = str(kws).strip()

            cls = obj.get("classifications", [])
            codes = []
            if isinstance(cls, list):
                for c in cls:
                    if isinstance(c, dict) and c.get("code"):
                        codes.append(str(c["code"]).strip().replace(" ", ""))
            codes = sorted(set([c for c in codes if c]))

            # 过滤
            if not codes or not (title or abstract):
                done_files.add(fp)
                continue

            text = f"{title} [SEP] {keywords} [SEP] {abstract}".strip()

            writer.writerow({
                "publication_number": pub,
                "text": text,
                "labels": json.dumps(codes, ensure_ascii=False),
                "n_labels": len(codes),
            })
            n_written += 1

            done_files.add(fp)

            # 定期落盘 progress，避免中断返工
            if len(done_files) % save_progress_every == 0:
                save_progress(progress_path, done_files)
                wf.flush()

            # 降低日志频率
            if idx % log_every == 0:
                print(f"Scanned files: {idx}/{len(txt_paths)} | written: {n_written} | done_files: {len(done_files)}")

    save_progress(progress_path, done_files)
    print(f"Done. total_written={n_written}, saved to {out_csv_path}")
    print(f"Progress saved to {progress_path}")

if __name__ == "__main__":
    build_multilabel_csv_from_txt_dir(
        txt_dir="/root/autodl-tmp/cn_use_1",
        out_csv_path="patents_multilabel_full.csv",
        progress_path="progress_done_files.json",
        max_docs=None,
        max_files=None,
        max_abstract_chars=1200,
        log_every=2000,
        save_progress_every=2000
    )
