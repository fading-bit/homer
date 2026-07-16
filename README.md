# homer-stylo

**Is "Homer" one hand or many?** A computational-stylometry pipeline that asks whether
the *Iliad* and *Odyssey* carry more internal stylistic variation than a single poet
produces — i.e. whether there is *more than one Homer within each poem*.

**Short answer from this project:** *yes in the weak sense, no in the strong sense.* Both
poems are significantly more internally variable than epics we know to be single-author
(Apollonius' *Argonautica*, Quintus' *Posthomerica*), and the effect survives controls for
length, register (narration only) and content (metre) — but the variation is **diffuse,
not discrete**: it does not resolve into a small number of separable, countable authors.
That fits the oral-traditional / evolutionary picture (many hands over generations) rather
than a committee with assigned books. See `RELATED_WORK.md` for how this sits beside prior
scholarship (Martindale & Tuffin 1996; Pavlopoulos & Konstantinidou 2022; Gorman & Gorman
2016), several of whose methods this project independently reconstructed.

---

## Install & data

Corpus: Perseus TEI (already fetched into `data/`). *Iliad* (Monro–Allen, 15,687 lines),
*Odyssey* (Murray, 12,107 lines); calibration epics in `data/raw/calib/`: Apollonius
*Argonautica* (5,834), Quintus *Posthomerica* (8,804), Hesiod *Theogony* + *Works & Days*
(1,042 + 831, treated jointly as a two-poem reference). `data/processed/lines.parquet` has
one row per verse line with three text columns: `text_raw` (verbatim), `text_dip`
(diplomatic — accents/case/elision kept; used by the syntax parser), `text_norm`
(lowercased, accents stripped, punctuation removed, ς→σ; the default for feature runs).

All Greek texts are from the [Perseus Digital Library](https://github.com/PerseusDL/canonical-greekLit)
(TEI XML), licensed CC BY-SA.

## Pipeline

| script | what it does | run |
|---|---|---|
| `helpers.py`, `corpus.py` | ingest EPUB/TEI → `lines.parquet` + language/quality report | `python src/corpus.py --config config.yaml` |
| `features_word.py`, `describe.py` | particles, lexical richness (**null**), formularity | `python src/features_word.py --config config.yaml` |
| `within_book.py` | sliding-window seam detection inside a book (char-trigram cosine + change points); `--voice {all,narration,speech}` | `python src/within_book.py --poem iliad --book 18 --voice narration` |
| `distances.py` | book×book Delta / char / NCD matrices, clustered heatmaps, MDS | `python src/distances.py --config config.yaml` |
| `speech_split.py` | narrator vs character-speech tagger (formula-based; ~50% speech) → `lines_voice.parquet` | `python src/speech_split.py --config config.yaml` |
| `calibrate.py` | **the headline test**: internal-dispersion of each work (Burrows's Delta over 400-line chunks) vs single-author epics; `--voice` for register control | `python src/calibrate.py --config config.yaml [--voice narration]` |
| `metre.py` | Greek hexameter scanner (92–95% scanned) → foot-pattern dispersion, **caesura** (penthemimeral/trochaic), spondaic-5th, Hermann's Bridge, bucolic diaeresis, enriched-metre dispersion | `python src/metre.py --config config.yaml` |
| `metre_localize.py` | content-free map of which **books** are metrically anomalous | `python src/metre_localize.py --config config.yaml` |
| `morphosyntax.py` | inflectional word-ending channel (content-light grammar proxy) | `python src/morphosyntax.py --config config.yaml` |
| `syntax.py` | **real dependency syntax** (Gorman sWord / POS / deprel) via a neural parser — see note below | `python src/syntax.py --config config.yaml` |
| `bootstrap.py` | 2000× chunk bootstrap → 95% CIs across word / metre / morphosyntax channels | `python src/bootstrap.py --config config.yaml` |
| `make_report.py` | assembles `outputs/homer_stylo_report.md` | `python src/make_report.py --config config.yaml` |

## Findings

- **Particles / lexical richness / formularity** (§4): particles separate the poems (habit,
  partly editorial); vocabulary richness is a **null** (flat across all 48 books);
  formularity is confounded with content.
- **Within-book seams** (§4.4): the detector re-discovers the **Shield of Achilles**
  (*Iliad* 18) and the **Catalogue of Ships** (*Iliad* 2); the Shield survives speech
  removal. Odyssey suspects (Second Nekyia, Catalogue of Heroines) are only faintly flagged.
- **Editions** (§4.5): char-level clustering splits the poems perfectly — an **edition
  artifact**; the word-level (Delta) split is weak. Within-poem structure is edition-robust.
- **Calibrated dispersion** (§6): Iliad 1.36, Odyssey 1.27 vs Apollonius 0.92, Quintus 1.08.
  **Narration-only** widens the gap (1.52 / 1.49). **Metre** (content-free) agrees (1.35 /
  1.35 vs 1.02 / 0.90); enriching metre with caesura/bridges leaves it intact. **Caesura**
  validates the scanner (Homer ~54% trochaic, the hand-counted figure) and shows drift to
  Quintus's 76%. **Morphosyntax** agrees but thinly (Odyssey marginal).
- **Bootstrap** (§6.3): the Iliad clears the single-author band on all three channels
  (P ≈ 1.00, 1.00, 0.997); the Odyssey on word + metre (P ≈ 1.00) and marginally on
  morphosyntax (P ≈ 0.85). The excess is not sampling noise.

## Reproducibility

The pipeline is deterministic (the bootstrap is seeded); same code + same data ⇒ identical
numbers. MDS scatter plots use unseeded layout, so those *figures* vary cosmetically. Built
with numpy 2.4, scikit-learn 1.8, pandas 3.0, scipy 1.17.

## Configuration

Everything result-affecting lives in `config.yaml`: file→poem mapping, book-header regexes,
line-cleaning patterns, and the two normalization variants. If a 24-book poem ingests as a
few giant books, fix a header pattern there — not the code.
