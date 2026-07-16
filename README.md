# homer-stylo

Computational EDA for the "is Homer one voice?" question. Turns EPUBs of the
*Iliad* and *Odyssey* (and anchor authors) into a tidy line-level table, then
— in later phases — into word-, character-, syntax-, and metre-level features
that triangulate on whether the 48 books read as one voice or many.

## Phase 0 — ingestion

```bash
python src/corpus.py --config config.yaml
```

Outputs:

| file | what |
|------|------|
| `data/processed/lines.parquet` / `.csv` | one row per verse line: `poem, role, book_label, book_no, line_no, text_raw, text_dip, text_norm` |
| `outputs/tables/ingest_report.md` | **the language/quality report** — per-source script, a Greek-vs-translation verdict, and per-book line counts (segmentation sanity check) |

The three text columns:
- **`text_raw`** — verbatim, as in the EPUB.
- **`text_dip`** — *diplomatic*: cleaned of line-number gutters and page junk, but
  accents, case, and elisions preserved. Use this for dialect/morphology-sensitive runs.
- **`text_norm`** — *aggressive*: lowercased, accents stripped, punctuation removed,
  final sigma folded (ς→σ). The default for most feature runs.

## Configuration

Everything that could change a result lives in `config.yaml`: which files map to
which poem, book-header regexes (defaults cover Greek-letter headers like
"Ἰλιάδος Α" and English "BOOK I"), line-cleaning patterns, and the two
normalization variants.
