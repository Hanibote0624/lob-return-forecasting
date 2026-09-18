#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import argparse
from pathlib import Path


DATE_DIR_RE = re.compile(r"^\d{8}$")
STEM_RE = re.compile(r"^(?P<prefix>.+?)_(?P<date>\d{8})_(?P<session>\d+)(?:_rename)?$")


def _open_text_with_fallback(path: Path, mode: str):
    """
    Try utf-8-sig then gbk (common on CN Windows exports).
    """
    try:
        return open(path, mode, encoding="utf-8-sig", newline="")
    except UnicodeDecodeError:
        return open(path, mode, encoding="gbk", newline="")


def derive_factor_col_base(txt_stem: str) -> str:
    """
    From e.g. 'gpmain_20250401_2_rename' -> 'gpmain'
    If pattern doesn't match, fallback to 'factor'.
    """
    m = STEM_RE.match(txt_stem)
    if not m:
        return "factor"
    return m.group("prefix")


def transform_output_filename(input_csv_name: str, stock_code: str) -> str:
    """
    Replace the source prefix with ``stock_code`` and remove ``_rename``.

    For example, ``gpmain_20250102_1_rename.csv`` becomes
    ``SAMPLE_20250102_1.csv`` when ``stock_code=SAMPLE``.
    """
    stem, ext = os.path.splitext(input_csv_name)
    stem = stem.replace("_rename", "")

    parts = stem.split("_")
    if parts and parts[0]:
        # 优先满足“前缀变成股票代码”
        parts[0] = str(stock_code)
        stem = "_".join(parts)
    else:
        stem = stem.replace("gpmain", str(stock_code), 1)

    return stem + ext


def iter_txt_rows(txt_path: Path):
    """
    Yield factor rows (list[str]) from txt. Skip empty lines.
    """
    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield line.split()  # whitespace split


def merge_one_pair(txt_path: Path, csv_path: Path, out_path: Path, overwrite: bool = False):
    if out_path.exists() and not overwrite:
        print(f"[SKIP] exists: {out_path}")
        return

    txt_stem = txt_path.stem
    col_base = derive_factor_col_base(txt_stem)

    txt_iter = iter_txt_rows(txt_path)

    # Peek first factor row to determine n_cols
    try:
        first_factor = next(txt_iter)
    except StopIteration:
        raise RuntimeError(f"TXT is empty: {txt_path}")

    n_factor_cols = len(first_factor)
    if n_factor_cols <= 0:
        raise RuntimeError(f"TXT first row has 0 cols: {txt_path}")

    if n_factor_cols == 1:
        new_cols = [col_base]
    else:
        new_cols = [f"{col_base}_{i}" for i in range(n_factor_cols)]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read CSV + write output streaming
    with _open_text_with_fallback(csv_path, "r") as fin, open(out_path, "w", encoding="utf-8-sig", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        try:
            header = next(reader)
        except StopIteration:
            raise RuntimeError(f"CSV is empty: {csv_path}")

        # Avoid header name conflicts
        header_set = set(header)
        final_new_cols = []
        for c in new_cols:
            cc = c
            k = 1
            while cc in header_set:
                cc = f"{c}_txt{k}"
                k += 1
            header_set.add(cc)
            final_new_cols.append(cc)

        writer.writerow(header + final_new_cols)

        # Write first data row using the peeked factor row
        data_rows = 0
        try:
            first_csv_row = next(reader)
        except StopIteration:
            # CSV has header only; should match no factor rows ideally
            raise RuntimeError(f"CSV has header only but TXT has data: {csv_path} vs {txt_path}")

        if len(first_factor) != n_factor_cols:
            raise RuntimeError(f"Inconsistent factor cols at first row: {txt_path}")

        writer.writerow(first_csv_row + first_factor)
        data_rows += 1

        # Remaining rows
        for csv_row in reader:
            try:
                factor_row = next(txt_iter)
            except StopIteration:
                raise RuntimeError(
                    f"Row mismatch: TXT ended early.\n"
                    f"  csv_path={csv_path}\n"
                    f"  txt_path={txt_path}\n"
                    f"  processed_csv_data_rows={data_rows}"
                )
            if len(factor_row) != n_factor_cols:
                raise RuntimeError(
                    f"Inconsistent factor cols in TXT.\n"
                    f"  txt_path={txt_path}\n"
                    f"  expected_cols={n_factor_cols}, got={len(factor_row)}\n"
                    f"  at_csv_data_row_index={data_rows+1}"
                )
            writer.writerow(csv_row + factor_row)
            data_rows += 1

        # Check TXT has no extra rows
        try:
            extra = next(txt_iter)
            raise RuntimeError(
                f"Row mismatch: TXT has extra rows after CSV ended.\n"
                f"  csv_path={csv_path}\n"
                f"  txt_path={txt_path}\n"
                f"  processed_csv_data_rows={data_rows}\n"
                f"  example_extra_factor_row={extra[:10]}"
            )
        except StopIteration:
            pass

    print(f"[OK] {csv_path.name} + {txt_path.name} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True, help="Root containing YYYYMMDD input directories")
    ap.add_argument("--output-root", required=True, help="Root for merged CSV output")
    ap.add_argument("--stock-code", required=True, help="Prefix used for output CSV filenames")
    ap.add_argument("--date", default=None, help="只处理某一天，例如 20250401；不填则处理所有日期目录")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出文件")
    args = ap.parse_args()

    in_root = Path(args.input_root)
    out_root = Path(args.output_root)
    stock_code = str(args.stock_code)

    if args.date is not None:
        date_dirs = [Path(args.date)]
    else:
        date_dirs = [Path(d.name) for d in in_root.iterdir() if d.is_dir() and DATE_DIR_RE.match(d.name)]

    date_dirs = sorted(date_dirs, key=lambda p: p.name)

    total = 0
    for d in date_dirs:
        in_dir = in_root / d.name
        if not in_dir.exists():
            print(f"[WARN] missing input date dir: {in_dir}")
            continue

        out_dir = out_root / d.name
        txt_files = sorted(in_dir.glob("*.txt"))
        if not txt_files:
            print(f"[WARN] no txt in {in_dir}")
            continue

        for txt_path in txt_files:
            csv_path = txt_path.with_suffix(".csv")
            if not csv_path.exists():
                print(f"[WARN] missing csv for: {txt_path.name}")
                continue

            out_name = transform_output_filename(csv_path.name, stock_code)
            out_path = out_dir / out_name

            merge_one_pair(txt_path, csv_path, out_path, overwrite=args.overwrite)
            total += 1

    print(f"[DONE] processed_pairs={total}")


if __name__ == "__main__":
    main()
