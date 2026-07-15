"""
corpus_tei.py — Phase 0 variant: ingest Perseus TEI XML.

Reads the canonical Perseus TEI XML files (Monro-Allen OCT edition) and
produces the same tidy line-level table as corpus.py:

    poem | role | book_label | book_no | line_no | text_raw | text_dip | text_norm

Run it:

    python src/corpus.py --config config.yaml

Or with explicit paths:

    python src/corpus.py \
        --iliad  data/raw/iliad_perseus.xml \
        --odyssey data/raw/odyssey_perseus.xml

Outputs:
    data/processed/lines.parquet / .csv
    outputs/tables/ingest_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from lxml import etree

# Re-use helpers from the EPUB ingester.
from helpers import (
    normalize_line,
    script_counts,
    label_script,
    build_report,
)

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def parse_tei(path: str | Path, poem: str, role: str = "homer") -> pd.DataFrame:
    """Parse a Perseus TEI XML file into a line-level DataFrame."""
    tree = etree.parse(str(path))
    rows: list[dict] = []
    for book_div in tree.xpath(
        "//tei:body/tei:div/tei:div[@n]", namespaces=TEI_NS
    ):
        book_no = int(book_div.get("n"))
        book_label = f"Book {book_no}"
        for l_elem in book_div.xpath(".//tei:l[@n]", namespaces=TEI_NS):
            line_no = int(l_elem.get("n"))
            text = "".join(l_elem.itertext()).strip()
            if not text:
                continue
            rows.append(
                {
                    "poem": poem,
                    "role": role,
                    "book_label": book_label,
                    "book_no": book_no,
                    "line_no": line_no,
                    "text_raw": text,
                    "text_dip": text,  # TEI is already clean
                }
            )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="homer-stylo Phase 0: TEI XML ingestion")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--iliad", default="data/raw/iliad_perseus.xml")
    ap.add_argument("--odyssey", default="data/raw/odyssey_perseus.xml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).resolve().parent
    norm_cfg = cfg["normalize"]["normalized"]

    frames: list[pd.DataFrame] = []
    for label, path in [("iliad", args.iliad), ("odyssey", args.odyssey)]:
        p = (root / path) if not Path(path).is_absolute() else Path(path)
        if not p.exists():
            print(f"[skip] {label}: not found at {p}", file=sys.stderr)
            continue
        print(f"[ingest] {label}  <-  {p.name}", file=sys.stderr)
        frames.append(parse_tei(p, label))

    if not frames:
        print("No TEI files found. Download them with:", file=sys.stderr)
        print(
            "  curl -sL https://raw.githubusercontent.com/PerseusDL/"
            "canonical-greekLit/master/data/tlg0012/tlg001/"
            "tlg0012.tlg001.perseus-grc2.xml -o data/raw/iliad_perseus.xml",
            file=sys.stderr,
        )
        print(
            "  curl -sL https://raw.githubusercontent.com/PerseusDL/"
            "canonical-greekLit/master/data/tlg0012/tlg002/"
            "tlg0012.tlg002.perseus-grc2.xml -o data/raw/odyssey_perseus.xml",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    df["text_norm"] = df["text_dip"].apply(lambda t: normalize_line(t, norm_cfg))

    proc = root / cfg["paths"]["processed_dir"]
    tables = root / cfg["paths"]["tables_dir"]
    proc.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    df.to_parquet(proc / "lines.parquet", index=False)
    df.to_csv(proc / "lines.csv", index=False)
    report = build_report(df, cfg)
    (tables / "ingest_report.md").write_text(report)

    print(f"\nWrote {len(df):,} lines → {proc / 'lines.parquet'}", file=sys.stderr)
    print(f"Report → {tables / 'ingest_report.md'}\n", file=sys.stderr)
    print(report)


if __name__ == "__main__":
    main()
