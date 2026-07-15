# homer-stylo

Computational EDA for the "is Homer one voice?" question. Turns EPUBs of the
*Iliad* and *Odyssey* (and anchor authors) into a tidy line-level table, then
— in later phases — into word-, character-, syntax-, and metre-level features
that triangulate on whether the 48 books read as one voice or many.

## Phase 0 — ingestion (this is what's built so far)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Put your EPUBs in data/raw/ and list them under `sources:` in config.yaml
# 2. Run:
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

## The one gate that matters

**The pipeline reads Greek. If your EPUBs are English translations, the report
will say so (Greek% near 0, a ⚠️ verdict).** Almost every planned feature —
particles, hexameter scansion, digamma observance, character n-grams over Greek
morphology — is Greek-specific and would otherwise measure the *translator's*
voice. Get the Ancient Greek text (Perseus / First1KGreek / a TLG-derived EPUB)
before going past EDA.

## Configuration

Everything that could change a result lives in `config.yaml`: which files map to
which poem, book-header regexes (defaults cover Greek-letter headers like
"Ἰλιάδος Α" and English "BOOK I"), line-cleaning patterns, and the two
normalization variants. If a poem that should have 24 even books shows up as 3
giant ones in the report, a header pattern needs adjusting — not the code.

## Roadmap (next phases, code not yet written)

1. word-level — particle profile, lexical richness, formularity index
2. character-level — n-grams, letter frequencies, 48×48 cross-entropy matrix
3. syntax-level — POS n-grams, dependency metrics (Perseus treebank), enjambement
4. metre — scansion (`greek_scansion`), foot shapes, caesura, bridges, SEDES
5. integration — three distance matrices, clustering, rolling stylometry
6. calibration — anchor-corpus projection, sanity checks, confound audit
7. narrator-vs-speech split; similes; type-scenes
