"""
Turns raw EPUB files into a tidy, line-level table:

    poem | role | book_label | book_no | line_no | text_raw | text_dip | text_norm

and — crucially — reports what *script* each source is in, so you find out
immediately whether you are analysing Homer's Greek or a translator's English.

Run it:

    python src/helpers.py --config config.yaml

# or point it at one file directly:
#     python src/helpers.py --input data/raw/iliad.epub --poem iliad

Outputs (per the paths in config.yaml):
    data/processed/lines.parquet   — the combined line table (all sources)
    data/processed/lines.csv       — same, human-readable
    outputs/tables/ingest_report.md — the language / quality report
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# ebooklib chatters warnings about the EPUB spec on many real-world files;
# they are harmless for text extraction.
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ebooklib")
warnings.filterwarnings("ignore", category=FutureWarning, module="ebooklib")
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
import ebooklib  # noqa: E402
from ebooklib import epub  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Script detection
# ─────────────────────────────────────────────────────────────────────────────

# Unicode ranges. Greek covers the basic Greek/Coptic block plus Greek Extended
# (polytonic accents live there), which is what any real Ancient Greek text uses.
_GREEK_RANGES = ((0x0370, 0x03FF), (0x1F00, 0x1FFF))
_LATIN_RANGES = ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F))


def _in_ranges(cp: int, ranges) -> bool:
    return any(lo <= cp <= hi for lo, hi in ranges)


def script_counts(text: str) -> dict:
    """Count alphabetic characters by script. Non-letters are ignored."""
    greek = latin = other = 0
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if _in_ranges(cp, _GREEK_RANGES):
            greek += 1
        elif _in_ranges(cp, _LATIN_RANGES):
            latin += 1
        else:
            other += 1
    total = greek + latin + other
    return {
        "greek": greek,
        "latin": latin,
        "other": other,
        "alpha_total": total,
        "greek_frac": greek / total if total else 0.0,
        "latin_frac": latin / total if total else 0.0,
    }


def label_script(counts: dict, cfg: dict) -> str:
    if counts["alpha_total"] == 0:
        return "empty"
    if counts["greek_frac"] >= cfg["greek_min_fraction"]:
        return "greek"
    if counts["latin_frac"] >= cfg["latin_min_fraction"]:
        return "latin"
    return "mixed"


# ─────────────────────────────────────────────────────────────────────────────
# EPUB → sections of text (spine order)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Section:
    order: int
    title: str
    text: str  # newline-separated lines, block structure preserved


def _html_to_lines(html: str) -> str:
    """
    Convert one XHTML document to text with poetry line breaks preserved.

    Verse EPUBs encode a line as a <p>, a <div>, or text ending in <br/>.
    We insert an explicit newline at every block boundary and every <br>, then
    let the caller split on newlines.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    # Turn <br> into newlines.
    for br in soup.find_all("br"):
        br.replace_with("\n")
    # Ensure block-level elements are newline-separated.
    block_tags = ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6",
                  "tr", "blockquote"]
    for tag in soup.find_all(block_tags):
        tag.append("\n")
    text = soup.get_text()
    # Normalise newlines; keep intra-line whitespace for now.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def _is_nav(item) -> bool:
    """True for the EPUB navigation document (EPUB3 nav or a nav-property item)."""
    if isinstance(item, epub.EpubNav):
        return True
    props = getattr(item, "properties", None) or []
    if "nav" in props:
        return True
    name = (item.get_name() or "").lower()
    return name.endswith("nav.xhtml") or name.endswith("toc.xhtml")


def _spine_documents(book: epub.EpubBook) -> list:
    """Content documents in reading order, nav excluded.

    Prefer the spine (true reading order); fall back to manifest document order.
    Spine entries come back as (idref, linear) tuples after a read round-trip,
    but may be bare ids or item objects, so handle all three.
    """
    items = []
    for entry in (book.spine or []):
        if isinstance(entry, tuple):
            idref = entry[0]
        elif isinstance(entry, str):
            idref = entry
        else:  # already an item object
            idref = getattr(entry, "id", None)
        it = book.get_item_with_id(idref) if idref else None
        if it is None or it.get_type() != ebooklib.ITEM_DOCUMENT or _is_nav(it):
            continue
        items.append(it)
    if not items:  # no usable spine — fall back
        items = [it for it in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
                 if not _is_nav(it)]
    return items


def read_epub_sections(path: str | Path) -> list[Section]:
    """Read an EPUB and return its content documents in reading (spine) order."""
    book = epub.read_epub(str(path))
    sections: list[Section] = []
    order = 0
    for item in _spine_documents(book):
        html = item.get_content().decode("utf-8", errors="replace")
        text = _html_to_lines(html)
        if not text.strip():
            continue
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        sections.append(Section(order=order, title=first[:80], text=text))
        order += 1
    return sections


# ─────────────────────────────────────────────────────────────────────────────
# Book segmentation
# ─────────────────────────────────────────────────────────────────────────────

# Classical 24-letter ordering used to number the books of each poem.
_GREEK_ALPHABET = list("αβγδεζηθικλμνξοπρστυφχψω")


def _greek_letter_to_num(label: str) -> int | None:
    base = strip_accents(label.lower()).strip()
    base = base.replace("ς", "σ")
    if len(base) == 1 and base in _GREEK_ALPHABET:
        return _GREEK_ALPHABET.index(base) + 1
    return None


_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def _roman_to_num(label: str) -> int | None:
    s = label.lower().strip()
    if not s or any(c not in _ROMAN for c in s):
        return None
    total, prev = 0, 0
    for c in reversed(s):
        val = _ROMAN[c]
        total += -val if val < prev else val
        prev = max(prev, val)
    return total or None


def _label_to_book_no(label: str) -> int | None:
    """Best-effort conversion of a captured header label to an integer."""
    if label is None:
        return None
    label = label.strip()
    if label.isdigit():
        return int(label)
    return _greek_letter_to_num(label) or _roman_to_num(label)


@dataclass
class RawBook:
    label: str
    book_no: int | None
    lines: list[str] = field(default_factory=list)


def segment_books(sections: list[Section], patterns: list[re.Pattern]) -> list[RawBook]:
    """
    Walk the spine-ordered text line by line, opening a new book whenever a
    header pattern fires. Text before the first header is a pseudo-book
    "front_matter" (dropped later unless it turns out to hold verse).
    """
    books: list[RawBook] = [RawBook(label="front_matter", book_no=0)]
    for sec in sections:
        for raw_line in sec.text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            matched_label = None
            for pat in patterns:
                m = pat.search(stripped)
                if m:
                    matched_label = m.group(1) if m.groups() else stripped
                    break
            if matched_label is not None:
                books.append(RawBook(
                    label=stripped[:40],
                    book_no=_label_to_book_no(matched_label),
                ))
            else:
                books[-1].lines.append(raw_line)
    return books


# ─────────────────────────────────────────────────────────────────────────────
# Line cleaning + normalization
# ─────────────────────────────────────────────────────────────────────────────

def strip_accents(text: str) -> str:
    """Remove combining diacritics (works for Greek polytonic and Latin)."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", stripped)


_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_line(text: str, cfg: dict) -> str:
    out = text
    if cfg.get("lowercase"):
        out = out.lower()
    if cfg.get("strip_accents"):
        out = strip_accents(out)
    if cfg.get("resolve_final_sigma"):
        out = out.replace("ς", "σ")
    if cfg.get("strip_punctuation"):
        out = _PUNCT_RE.sub(" ", out)
    if cfg.get("collapse_whitespace"):
        out = _WS_RE.sub(" ", out).strip()
    return out


def clean_line(raw: str, drop_pats, strip_pats, min_chars: int) -> str | None:
    """Return a cleaned verse line, or None if the line is noise."""
    line = raw.strip()
    if not line:
        return None
    for pat in drop_pats:
        if pat.search(line):
            return None
    for pat in strip_pats:
        line = pat.sub("", line, count=1)
    line = line.strip()
    if len(line) < min_chars:
        return None
    return line


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def _compile(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, flags=re.IGNORECASE | re.UNICODE) for p in patterns]


def ingest_source(path: str, poem: str, role: str, cfg: dict) -> pd.DataFrame:
    """Ingest one EPUB into a line-level DataFrame for that poem."""
    ing = cfg["ingest"]
    header_pats = _compile(ing["book_header_patterns"])
    drop_pats = _compile(ing["drop_line_patterns"])
    strip_pats = _compile(ing["strip_leading_patterns"])
    norm_cfg = cfg["normalize"]["normalized"]

    sections = read_epub_sections(path)
    if not sections:
        raise ValueError(f"No readable text found in {path}")

    raw_books = segment_books(sections, header_pats)

    rows = []
    for rb in raw_books:
        line_no = 0
        for raw in rb.lines:
            cleaned = clean_line(raw, drop_pats, strip_pats, ing["min_line_chars"])
            if cleaned is None:
                continue
            line_no += 1
            rows.append({
                "poem": poem,
                "role": role,
                "book_label": rb.label,
                "book_no": rb.book_no,
                "line_no": line_no,
                "text_raw": raw.strip(),
                "text_dip": cleaned,                      # diplomatic (cleaned only)
                "text_norm": normalize_line(cleaned, norm_cfg),  # aggressive
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            f"{path}: text was read but no verse lines survived cleaning. "
            "Check book_header_patterns / drop_line_patterns in config.yaml."
        )
    # Drop a front_matter pseudo-book only if real books were also found.
    real = df[df["book_label"] != "front_matter"]
    if not real.empty:
        df = real.reset_index(drop=True)
    return df


def build_report(df: pd.DataFrame, cfg: dict) -> str:
    """A markdown language/quality report — the real Phase-0 deliverable."""
    sd_cfg = cfg["script_detection"]
    lines = ["# Ingestion report\n"]

    # Per-source script + size.
    lines.append("## Sources: script & size\n")
    lines.append("| poem | role | books | lines | tokens (norm) | script | greek% |")
    lines.append("|------|------|------:|------:|--------------:|--------|-------:|")
    overall_greek = overall_alpha = 0
    for poem, g in df.groupby("poem"):
        counts = script_counts(" ".join(g["text_raw"]))
        overall_greek += counts["greek"]
        overall_alpha += counts["alpha_total"]
        n_tokens = g["text_norm"].str.split().map(len).sum()
        lines.append(
            f"| {poem} | {g['role'].iloc[0]} | {g['book_no'].nunique()} "
            f"| {len(g):,} | {int(n_tokens):,} "
            f"| {label_script(counts, sd_cfg)} | {counts['greek_frac']*100:4.1f} |"
        )

    # The headline verdict.
    corpus_greek_frac = overall_greek / overall_alpha if overall_alpha else 0.0
    lines.append("\n## Verdict\n")
    if corpus_greek_frac < sd_cfg["translation_warning_below_greek"]:
        lines.append(
            f"> ⚠️  **The corpus is {corpus_greek_frac*100:.1f}% Greek.** "
            "This looks like a **translation**, not the Greek text. The "
            "particle/meter/digamma/character-morphology features are all "
            "Greek-specific and will measure the *translator's* voice, not "
            "Homer's. Get the Ancient Greek text before proceeding past EDA.\n"
        )
    elif corpus_greek_frac >= sd_cfg["greek_min_fraction"]:
        lines.append(
            f"> ✅  Corpus is **{corpus_greek_frac*100:.1f}% Greek** — the full "
            "Greek-specific feature stack is valid.\n"
        )
    else:
        lines.append(
            f"> ⚠️  Corpus is **{corpus_greek_frac*100:.1f}% Greek** (mixed). "
            "Possibly a bilingual/parallel edition — inspect per-book scripts "
            "below before trusting anything.\n"
        )

    # Per-book line counts (a fast way to see mis-segmentation: a poem that
    # should have 24 roughly-even books but shows 3 giant ones is a header bug).
    lines.append("## Lines per book (segmentation sanity check)\n")
    for poem, g in df.groupby("poem"):
        per = g.groupby(["book_no", "book_label"]).size()
        lines.append(f"\n**{poem}** — {g['book_no'].nunique()} books detected:\n")
        lines.append("```")
        for (bno, blabel), n in per.items():
            lines.append(f"  book {str(bno):>3}  ({blabel[:28]:<28})  {n:>5} lines")
        lines.append("```")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="homer-stylo Phase 0: EPUB ingestion")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", help="single EPUB path (bypasses config sources)")
    ap.add_argument("--poem", help="poem label when using --input")
    ap.add_argument("--role", default="homer", help="homer|anchor when using --input")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).resolve().parent

    if args.input:
        jobs = [(args.input, args.poem or Path(args.input).stem, args.role)]
    else:
        jobs = [(s["path"], s["poem"], s.get("role", "homer"))
                for s in cfg["sources"]]

    frames = []
    for path, poem, role in jobs:
        p = (root / path) if not Path(path).is_absolute() else Path(path)
        if not p.exists():
            print(f"[skip] {poem}: file not found at {p}", file=sys.stderr)
            continue
        print(f"[ingest] {poem}  <-  {p.name}", file=sys.stderr)
        frames.append(ingest_source(str(p), poem, role, cfg))

    if not frames:
        print("No sources ingested. Put your EPUBs in data/raw/ and list them "
              "under `sources:` in config.yaml (or pass --input).", file=sys.stderr)
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)

    proc = root / cfg["paths"]["processed_dir"]
    tables = root / cfg["paths"]["tables_dir"]
    proc.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    df.to_parquet(proc / "lines.parquet", index=False)
    df.to_csv(proc / "lines.csv", index=False)
    report = build_report(df, cfg)
    (tables / "ingest_report.md").write_text(report)

    print(f"\nWrote {len(df):,} lines → {proc/'lines.parquet'}", file=sys.stderr)
    print(f"Report → {tables/'ingest_report.md'}\n", file=sys.stderr)
    print(report)


if __name__ == "__main__":
    main()
