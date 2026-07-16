"""
elision.py — elision-aware normalization for Greek verse.

In the hexameter a final short vowel is routinely dropped before a following
vowel and marked with an apostrophe: δ' for δέ, ἀλλ' for ἀλλά, μ' for με. The
default normalizer strips punctuation, which turns these into bare stems (δ, ἀλλ,
μ) — inflating the "vocabulary", splitting particle counts across a real and a
truncated form, and confusing the scansion. This module restores the commonest
elided forms *before* normalization, so tokens land on their full lemma.

Restoration is not always unique (τ' is almost always τε, but μ' could be με or
μοι, δ' could be δέ or δή); we restore only the dominant reading and leave genuinely
ambiguous stems alone. Use `restore(text_dip)` on the accented/diplomatic text,
then normalize as usual.

Integration: apply in corpus_tei.py before normalize_line, or add a boolean
`restore_elision` to config.yaml and gate it there.
"""

from __future__ import annotations

import re

APOS = "['\u2019\u02bc]"  # straight ', right single quote, modifier apostrophe

# base (lowercase, accented) elided form -> full form. Prepositions/conjunctions/
# particles that elide constantly in Homer; only near-unambiguous cases.
_MAP = {
    "δ": "δέ", "τ": "τε", "γ": "γε", "ῥ": "ῥα", "ἄρ": "ἄρα", "ἄλλ": "ἄλλα",
    "ἀλλ": "ἀλλά", "οὐδ": "οὐδέ", "μηδ": "μηδέ", "ἀπ": "ἀπό", "ὑπ": "ὑπό",
    "ἐπ": "ἐπί", "κατ": "κατά", "μετ": "μετά", "παρ": "παρά", "ἀν": "ἀνά",
    "ἀμφ": "ἀμφί", "δι": "διά", "ποτ": "ποτί", "ὑφ": "ὑπό", "ἀφ": "ἀπό",
    "ἐφ": "ἐπί", "καθ": "κατά", "μεθ": "μετά", "ἐπ᾽": "ἐπί",
    "μ": "με", "σ": "σε", "τοῦτ": "τοῦτο", "ταῦτ": "ταῦτα", "οὔτ": "οὔτε",
    "μήτ": "μήτε", "ἠδ": "ἠδέ", " οὐκ": "οὐκ",
}
# also handle rough-breathing 'θ' spellings that stand for τε before an aspirate
_MAP["θ"] = "τε"


def _restore_token(tok: str) -> str:
    core = re.sub(APOS + r"$", "", tok)
    low = core.lower()
    if low in _MAP:
        full = _MAP[low]
        # preserve an initial capital (line-initial words)
        if core[:1].isupper():
            full = full[:1].upper() + full[1:]
        return full
    return tok


def restore(text: str) -> str:
    """Restore elided forms in an accented line. Idempotent on non-elided tokens."""
    out = []
    for tok in text.split():
        out.append(_restore_token(tok) if re.search(APOS + r"$", tok) else tok)
    return " ".join(out)


if __name__ == "__main__":
    # quick effectiveness check on the corpus
    import sys
    from pathlib import Path
    import yaml
    import pandas as pd

    cfg = yaml.safe_load(Path(sys.argv[sys.argv.index("--config") + 1]
                              if "--config" in sys.argv else "config.yaml").read_text())
    root = Path("config.yaml").resolve().parent
    df = pd.read_parquet(root / cfg["paths"]["processed_dir"] / "lines.parquet")
    tot = fixed = 0
    examples = []
    for t in df.text_dip:
        for tok in str(t).split():
            if re.search(APOS + r"$", tok):
                tot += 1
                r = _restore_token(tok)
                if r != tok:
                    fixed += 1
                    if len(examples) < 12:
                        examples.append(f"{tok} → {r}")
    print(f"elided (apostrophe-final) tokens: {tot:,}")
    print(f"restored to a full form:          {fixed:,} ({fixed/max(tot,1):.0%})")
    print("examples:", ", ".join(examples))
