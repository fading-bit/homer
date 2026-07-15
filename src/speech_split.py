"""
speech_split.py — the narrator vs. character-speech split.

Homer alternates between the poet's narration and quoted character speech, and
that alternation is a REGISTER shift that contaminates any search for authorial
seams. This module tags every line as narration ('N') or direct speech ('S') so
the two strata can be analysed separately — the key move for asking whether
there is "more than one Homer" without being fooled by "a character started
talking".

Method (no quotation marks exist in the text): speeches are bounded by Homer's
formulaic frames. An INTRODUCTION formula (e.g. τὸν δ' ἀπαμειβόμενος προσέφη,
… ἔπεα πτερόεντα προσηύδα) opens a speech on the following line; a RESUMPTION
formula (ὣς φάτο, ὣς εἰπών, ἦ ῥα) returns to narration. This has high precision
on formulaic frames but under-counts speeches opened by a plain lexical verb, so
the speech share is a lower bound — stated, not hidden.

Outputs:
  data/processed/lines_voice.parquet   (adds a 'voice' column: 'N' | 'S')
  outputs/tables/voice_breakdown.csv    (overall narration/speech counts)
  outputs/tables/voice_markers.csv      (words most typical of speech vs narration)
  outputs/figures/voice_markers.png     (the register contrast)

Run:  python src/speech_split.py --config config.yaml
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from viz import savefig

# Speech-INTRODUCTION cues (substring match on normalised text). Presence marks
# the line as a narration frame whose speech begins on the NEXT line. This list
# is deliberately restricted to reliable reply/address frames, which pair with a
# resumption formula and so yield clean speech spans. Speeches opened by a plain
# lexical verb (λίσσετο "besought", ἠρᾶτο "prayed", μῦθον ἔτελλε "commanded") are
# therefore missed and stay labelled narration — a precision-over-recall choice.
INTRO_CUES = [
    "προσεφη", "προσεειπ", "προσειπ", "προσηυδ", "προσεφων",
    "αμειβ", "ημειβ", "μετεφη", "μετηυδ", "μετεειπ",
    "επεα πτεροεντα", "μυθον εειπ",
]

# Speech-RESUMPTION cues (line START), marking a return to narration.
RESUMPTION_RE = re.compile(
    r"^(ωσ (φατο|εφατ|εφατο|φαθ|εειπ|ειπων|ειπουσα|ειπουσ|φαμενη|φαμενοσ|αρα|"
    r"φατ|φωνησασ)|η ρα|η ρ |η,|ητοι ο |ητοι ογ|ωσ ειπ|ωσ φατ)"
)


def is_intro(text_norm: str) -> bool:
    return any(cue in text_norm for cue in INTRO_CUES)


def is_resumption(text_norm: str) -> bool:
    return bool(RESUMPTION_RE.match(text_norm))


def tag_voice(df: pd.DataFrame) -> pd.Series:
    """Walk each book in line order; return a 'voice' label per row ('N'/'S')."""
    voice = np.empty(len(df), dtype=object)
    idx = 0
    for _poem, g in df.groupby(["poem", "book_no"], sort=False):
        in_speech = False
        for _, row in g.iterrows():
            t = row.text_norm
            intro = is_intro(t)
            if in_speech:
                if is_resumption(t):
                    voice[idx] = "N"          # narrator resumes on this line
                    in_speech = intro          # a new speech may open immediately
                elif intro:
                    voice[idx] = "N"          # a reply frame: narration line that
                    in_speech = True           # closes this speech and opens the next
                else:
                    voice[idx] = "S"
            else:
                voice[idx] = "N"               # narration (incl. the intro frame)
                if intro:
                    in_speech = True
            idx += 1
    return pd.Series(voice, index=df.index, name="voice")


# ── Register contrast: what marks speech vs narration ───────────────────────

def voice_markers(df: pd.DataFrame, min_count=30, top_n=14):
    narr = Counter(" ".join(df[df.voice == "N"].text_norm).split())
    spch = Counter(" ".join(df[df.voice == "S"].text_norm).split())
    Nn, Ns = sum(narr.values()), sum(spch.values())
    rows = []
    for w in set(narr) | set(spch):
        a, b = spch.get(w, 0), narr.get(w, 0)
        if a + b < min_count:
            continue
        fs, fn = a / Ns, b / Nn
        lr = np.log2(((a + 0.5) / Ns) / ((b + 0.5) / Nn))
        rows.append({"word": w, "freq_speech_pm": fs * 1000,
                     "freq_narr_pm": fn * 1000, "log2_ratio_speech_over_narr": lr,
                     "count": a + b})
    m = pd.DataFrame(rows).sort_values("log2_ratio_speech_over_narr")
    speechy = m.tail(top_n).iloc[::-1]
    narry = m.head(top_n)
    return m, speechy, narry


def plot_markers(speechy, narry, figures_dir):
    fig, ax = plt.subplots(figsize=(11, 8))
    data = pd.concat([narry.assign(side="narration"),
                      speechy.assign(side="speech")])
    data = data.sort_values("log2_ratio_speech_over_narr")
    colors = ["#b5651d" if v < 0 else "#4a7c59" for v in data.log2_ratio_speech_over_narr]
    ax.barh(range(len(data)), data.log2_ratio_speech_over_narr, color=colors)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.word, fontsize=10)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("log2 frequency ratio  (left = typical of narration, "
                  "right = typical of speech)")
    ax.set_title("What distinguishes character speech from narration\n"
                 "(most discriminating frequent words, whole corpus)", fontsize=12)
    ax.text(0.98, 0.98, "SPEECH", transform=ax.transAxes, color="#4a7c59",
            fontsize=11, fontweight="bold", va="top", ha="right")
    ax.text(0.02, 0.02, "NARRATION", transform=ax.transAxes, color="#b5651d",
            fontsize=11, fontweight="bold", va="bottom", ha="left")
    return savefig(fig, figures_dir, "voice_markers")


# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).resolve().parent
    pdir = root / cfg["paths"]["processed_dir"]
    tables = root / cfg["paths"]["tables_dir"]
    figures = root / cfg["paths"]["figures_dir"]
    for d in (tables, figures):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(pdir / "lines.parquet")
    df["voice"] = tag_voice(df)
    df.to_parquet(pdir / "lines_voice.parquet")

    n_spch = int((df.voice == "S").sum())
    n_narr = int((df.voice == "N").sum())
    tok_spch = int(df[df.voice == "S"].text_norm.str.split().map(len).sum())
    tok_narr = int(df[df.voice == "N"].text_norm.str.split().map(len).sum())
    pd.DataFrame([
        {"stratum": "narration", "lines": n_narr, "tokens": tok_narr},
        {"stratum": "speech", "lines": n_spch, "tokens": tok_spch},
    ]).to_csv(tables / "voice_breakdown.csv", index=False)

    m, speechy, narry = voice_markers(df)
    m.to_csv(tables / "voice_markers.csv", index=False)
    plot_markers(speechy, narry, figures)

    # console: proportion + a spot-check of Iliad 1
    print("=== NARRATOR vs. CHARACTER-SPEECH SPLIT ===\n")
    print(f"lines : {n_narr:,} narration + {n_spch:,} speech "
          f"({100*n_spch/(n_narr+n_spch):.1f}% speech)")
    print(f"tokens: {tok_narr:,} narration + {tok_spch:,} speech "
          f"({100*tok_spch/(tok_narr+tok_spch):.1f}% speech)")
    print("\nMost speech-typical function words (log2 ratio speech/narr):")
    for _, r in speechy.head(8).iterrows():
        print(f"  {r.word:12s} {r.log2_ratio_speech_over_narr:+.2f}")
    print("\nMost narration-typical:")
    for _, r in narry.head(8).iterrows():
        print(f"  {r.word:12s} {r.log2_ratio_speech_over_narr:+.2f}")

    print("\nSpot-check A — Iliad 1, lines 1-25 (proem + lexically-opened speeches "
          "the tagger misses):")
    il1 = df[(df.poem == "iliad") & (df.book_no == 1)].sort_values("line_no")
    for _, r in il1[il1.line_no <= 25].iterrows():
        print(f"  {r.line_no:3d} {r.voice}  {r.text_raw[:60]}")
    print("\nSpot-check B — Iliad 1, lines 120-145 (reply-frame dialogue, "
          "tagged correctly):")
    for _, r in il1[(il1.line_no >= 120) & (il1.line_no <= 145)].iterrows():
        print(f"  {r.line_no:3d} {r.voice}  {r.text_raw[:60]}")

    print(f"\nWrote {pdir/'lines_voice.parquet'}, voice tables + figure")


if __name__ == "__main__":
    main()
