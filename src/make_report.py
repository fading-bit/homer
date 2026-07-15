"""
make_report.py — build the project report PDF (Phases 0–2b).

Reads outputs/tables/*.csv and outputs/figures/*.png (from describe.py,
features_word.py, and within_book.py) and assembles a Greek-capable PDF at
outputs/report/homer_stylo_report.pdf.

Run:  python src/make_report.py --config config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

# ── Fonts (DejaVu Sans covers polytonic Greek) ──────────────────────────────
from matplotlib.font_manager import findfont, FontProperties
_FD = Path(findfont(FontProperties(family="DejaVu Sans"))).parent
pdfmetrics.registerFont(TTFont("DJ", str(_FD / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DJ-B", str(_FD / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DJ-I", str(_FD / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DJ", normal="DJ", bold="DJ-B", italic="DJ-I")

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#666666")
IL = colors.HexColor("#b5651d")
OD = colors.HexColor("#1d6fb5")
RULE = colors.HexColor("#cccccc")

styles = getSampleStyleSheet()


def S(name, **kw):
    base = kw.pop("parent", styles["Normal"])
    kw.setdefault("fontName", "DJ")
    kw.setdefault("textColor", INK)
    return ParagraphStyle(name, parent=base, **kw)


ST = {
    "title": S("title", fontName="DJ-B", fontSize=19, leading=23, spaceAfter=4),
    "subtitle": S("subtitle", fontName="DJ", fontSize=12, leading=15, textColor=MUTED, spaceAfter=2),
    "byline": S("byline", fontName="DJ", fontSize=9.5, leading=13, textColor=MUTED),
    "h1": S("h1", fontName="DJ-B", fontSize=13.5, leading=17, spaceBefore=16, spaceAfter=5),
    "h2": S("h2", fontName="DJ-B", fontSize=11, leading=14, spaceBefore=10, spaceAfter=3),
    "body": S("body", fontSize=9.7, leading=13.8, alignment=TA_JUSTIFY, spaceAfter=6),
    "abstract": S("abstract", fontSize=9.5, leading=13.5, alignment=TA_JUSTIFY,
                  leftIndent=10, rightIndent=10, textColor=colors.HexColor("#333333")),
    "caption": S("caption", fontName="DJ-I", fontSize=8.3, leading=11, textColor=MUTED,
                 alignment=TA_CENTER, spaceBefore=3, spaceAfter=10),
    "cell": S("cell", fontSize=8.2, leading=10.5),
    "cellr": S("cellr", fontSize=8.2, leading=10.5, alignment=2),
    "cellh": S("cellh", fontName="DJ-B", fontSize=8.2, leading=10.5, textColor=colors.white),
    "num": S("num", fontSize=9.7, leading=13.8, alignment=TA_JUSTIFY, spaceAfter=6, leftIndent=16, firstLineIndent=-16),
    "ref": S("ref", fontSize=8.6, leading=11.5, leftIndent=14, firstLineIndent=-14, spaceAfter=3),
}

CONTENT_W = A4[0] - 4 * cm


def P(text, style="body"):
    return Paragraph(text, ST[style])


def fig(path: Path, caption: str, max_w=CONTENT_W, max_h=20 * cm):
    im = PILImage.open(path)
    w, h = im.size
    scale = min(max_w / w, max_h / h)
    return [Image(str(path), width=w * scale, height=h * scale), P(caption, "caption")]


def table(headers, rows, col_widths, aligns=None, header_fill=INK):
    aligns = aligns or ["L"] * len(headers)
    head = [Paragraph(h, ST["cellh"]) for h in headers]
    body = []
    for r in rows:
        body.append([Paragraph(str(v), ST["cellr"] if a == "R" else ST["cell"])
                     for v, a in zip(r, aligns)])
    t = Table([head] + body, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_fill),
        ("FONTNAME", (0, 0), (-1, -1), "DJ"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f6f2")]),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, RULE),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
    ]))
    return t


# ────────────────────────────────────────────────────────────────────────────

def build(cfg, root):
    tdir = root / cfg["paths"]["tables_dir"]
    fdir = root / cfg["paths"]["figures_dir"]
    pdir = root / cfg["paths"]["processed_dir"]

    df = pd.read_parquet(pdir / "lines.parquet")
    parts = pd.read_csv(tdir / "particles_freq_permille.csv", index_col=0)
    form = pd.read_csv(tdir / "formularity.csv")
    rich = pd.read_csv(tdir / "lexical_richness.csv")
    het = pd.read_csv(tdir / "within_book_heterogeneity.csv")

    il = df[df.poem == "iliad"]; od = df[df.poem == "odyssey"]
    il_tok = int(il.text_norm.str.split().map(len).sum())
    od_tok = int(od.text_norm.str.split().map(len).sum())

    story = []

    # ── Title ──
    story += [
        P("Is Homer One Voice?", "title"),
        P("A computational-stylometric investigation of the Iliad and Odyssey", "subtitle"),
        Spacer(1, 6),
        P("Project report · Phases 0–2b (corpus, word-level EDA, within-book analysis)"
          "<br/>Fada · July 2026", "byline"),
        Spacer(1, 4),
        Table([[""]], colWidths=[CONTENT_W],
              style=TableStyle([("LINEBELOW", (0, 0), (-1, -1), 1, INK)])),
        Spacer(1, 10),
    ]

    # ── Abstract ──
    story += [
        P("Abstract", "h2"),
        P("This project asks whether the <i>Iliad</i> and <i>Odyssey</i> read as the "
          "product of a single poetic voice or of many, using quantitative analysis of "
          "the Greek text. The working corpus is the full text of both poems "
          f"({len(df):,} verse lines, ~{(il_tok+od_tok)//1000}k tokens) obtained as "
          "canonical TEI XML from the Perseus Digital Library. This report covers the "
          "data pipeline, the word-level layer of exploratory analysis, and a "
          "within-book analysis that looks for stylistic seams inside a single book. "
          "Three results stand out. (i) A book-by-book <b>particle profile separates "
          "the two poems</b>, the most promising voice-type signal, though it is now "
          "qualified by an edition caveat (below). (ii) <b>Formularity</b> also "
          "separates the poems but is confounded with content. (iii) A within-book "
          "seam detector <b>independently rediscovers the two most famous embedded "
          "set-pieces of the Iliad</b> — the Shield of Achilles and the Catalogue of "
          "Ships — as the passages most stylistically distinct from their surroundings. "
          "All findings are exploratory: a verdict on one-versus-many awaits inter-book "
          "distance modelling and calibration against known single-author verse, set "
          "out in the plan.", "abstract"),
        Spacer(1, 6),
    ]

    # ── 1. Question & design ──
    story += [
        P("1&nbsp;&nbsp;Research question and design", "h1"),
        P("The \u201cHomeric Question\u201d \u2014 whether one author composed the "
          "<i>Iliad</i> and <i>Odyssey</i> \u2014 is really several questions at "
          "different levels. At the level of composition, Parry and Lord's "
          "oral-formulaic theory recast the poems as the sediment of a tradition of "
          "many singers; at the level of discourse, Bakker reads Homeric verse as "
          "structured speech; and at the level of the text itself, Nagy's evolutionary "
          "model denies there was ever a single fixed original. \u201cNot one "
          "voice\u201d can therefore mean several distinct things, and a computational "
          "study must be clear about which it is probing.", "body"),
        P("The central hazard is that most measurable differences between stretches of "
          "Homer track <i>content</i> \u2014 what a passage is about \u2014 rather than "
          "authorial <i>habit</i>. A battle scene and a divine council differ in "
          "vocabulary no matter who wrote them. The project is therefore organised "
          "around one discipline: every feature is judged by whether it measures "
          "content (confounded with the story) or habit (a candidate voice signal), "
          "and analysis triangulates across independent feature families \u2014 word, "
          "character, syntax, metre \u2014 so a boundary counts as real only when "
          "signals that ought to be independent agree on it. This report covers the "
          "word-level family, plus a first move to finer, within-book resolution.", "body"),
    ]

    # ── 2. Corpus ──
    story += [
        P("2&nbsp;&nbsp;Corpus and source", "h1"),
        P("Our source is the digital Greek text of both poems from the <b>Perseus "
          "Digital Library</b>, parsed from canonical TEI XML whose explicit book and "
          "line markup makes segmentation exact rather than inferred. A Phase-0 "
          "ingester turns the XML into a tidy line-level table with three text columns "
          "\u2014 verbatim, a lightly cleaned <i>diplomatic</i> form, and a "
          "<i>normalised</i> form (lower-cased, accents and punctuation removed, "
          "elision marks stripped, final sigma folded) \u2014 and a script check "
          "confirms the corpus is 98% Greek. The edition and every normalisation "
          "switch live in a config file, so runs are reproducible.", "body"),
        P("One caveat must be stated up front, because it bears on the cross-poem "
          "comparisons below: the two poems come from <b>different editors</b>. The "
          "Iliad is Monro &amp; Allen's Oxford Classical Text; the Odyssey is Murray's "
          "edition. Editors differ in orthographic conventions (elision, movable-nu, "
          "accentuation) and in which lines they admit, so some apparent difference "
          "<i>between</i> the poems may be editorial rather than authorial. This does "
          "not affect the within-book analysis (which compares passages inside one "
          "poem, one edition); it is revisited in \u00a76 and is the first task of the "
          "plan.", "body"),
        table(
            ["Poem", "Edition (via Perseus)", "Books", "Lines", "Tokens"],
            [["Iliad", "Monro & Allen (OCT)", "24", f"{len(il):,}", f"{il_tok:,}"],
             ["Odyssey", "Murray", "24", f"{len(od):,}", f"{od_tok:,}"],
             ["Total", "\u2014", "48", f"{len(df):,}", f"{il_tok+od_tok:,}"]],
            [2.4*cm, 5.0*cm, 1.7*cm, 2.6*cm, 2.6*cm],
            aligns=["L", "L", "R", "R", "R"],
        ),
        P("Table 1. Working corpus. Note the two poems come from different editions "
          "(a limitation, \u00a76).", "caption"),
    ]
    s = il[il.book_no == 1].head(2)
    srows = []
    for _, r in s.iterrows():
        srows.append([f"Il. 1.{r.line_no}", r.text_raw])
        srows.append(["\u2003(normalised)", r.text_norm])
    story += [
        table(["Ref.", "Line"], srows, [2.4*cm, CONTENT_W - 2.4*cm], aligns=["L", "L"]),
        P("Table 2. Verbatim and normalised forms of a line (Iliad 1.1\u20131.2); "
          "features run on the normalised column.", "caption"),
    ]

    # ── 3. Method ──
    story += [
        P("3&nbsp;&nbsp;Method", "h1"),
        P("At the <b>book level</b>, four analyses run over the normalised tokens of "
          "each of the 48 books. (1) A <b>particle / function-word profile</b>: "
          "relative frequencies (per mille) of ~30 Homeric particles and connectives, "
          "z-scored across books; because these words carry almost no subject matter, "
          "variation in them is a candidate habit signal. (2) <b>Lexical richness</b> "
          "via length-robust measures (Yule's K, the moving-average type\u2013token "
          "ratio MATTR, the hapax rate). (3) <b>Formularity</b>: the share of a book's "
          "lines that recur elsewhere in the corpus, plus repeated n-gram density. "
          "(4) A most-frequent-word matrix, stored for the later distance phase.", "body"),
        P("At the <b>within-book level</b>, a sliding window (60 lines, step 10) moves "
          "through a single book; each window is described by its character-trigram "
          "profile (content-lighter than words and stable in short spans). Three "
          "readings follow: the cosine distance of each window to the book's own "
          "centroid (a rolling divergence curve), a window\u00d7window self-distance "
          "matrix (block structure signals a seam), and change points detected with a "
          "penalised search (PELT) mapped back to line numbers. This first pass uses "
          "surface word forms only; lemmatisation, syntax, and metre enter later.", "body"),
    ]

    # ── 4. Results ──
    story += [P("4&nbsp;&nbsp;Results", "h1")]

    # 4.1 particles
    p_il = parts[parts.poem == "iliad"].drop(columns="poem").mean()
    p_od = parts[parts.poem == "odyssey"].drop(columns="poem").mean()
    diff = (p_il - p_od).sort_values()
    prows = []
    for name in list(diff.index[:4]) + list(diff.index[-4:]):
        d = p_il[name] - p_od[name]
        prows.append([name, f"{p_il[name]:.2f}", f"{p_od[name]:.2f}", f"{d:+.2f}",
                      "Iliad" if d > 0 else "Odyssey"])
    story += [
        P("4.1&nbsp;&nbsp;Particle profile \u2014 the promising signal", "h2"),
        P("The two poems separate on function-word usage: <i>\u03b4\u03ad</i> is "
          "markedly commoner in the Iliad, while <i>\u03c4\u03bf\u03b9</i>, "
          "<i>\u03ba\u03b1\u03af</i>, <i>\u03b1\u1f50\u03c4\u03ac\u03c1</i> and "
          "<i>\u1f22</i> lean to the Odyssey, and <i>\u1f24\u03c4\u03bf\u03b9</i> is "
          "effectively Iliad-only. Because these are content-independent words, this "
          "is a genuine stylistic-habit signal rather than an artefact of subject "
          "matter; the block difference is visible by eye in Figure&nbsp;1 (most "
          "cleanly in the \u1f24\u03c4\u03bf\u03b9 column, uniformly low across the "
          "Odyssey). Two caveats: this separates the two <i>poems</i> and says nothing "
          "yet about single-versus-multiple authorship <i>within</i> a poem; and, given "
          "\u00a72, part of the difference could reflect the two editors' conventions "
          "rather than Homer, which the matched-edition step in the plan will test.", "body"),
        table(["Particle", "Iliad (\u2030)", "Odyssey (\u2030)", "\u0394", "Leans"],
              prows, [3.0*cm, 2.8*cm, 2.8*cm, 2.2*cm, 2.6*cm],
              aligns=["L", "R", "R", "R", "L"]),
        P("Table 3. Largest Iliad\u2013Odyssey divergences in particle frequency "
          "(per mille).", "caption"),
    ]
    story += fig(fdir / "particle_heatmap.png",
                 "Figure 1. Particle / function-word profile by book (per-mille, "
                 "z-scored). Iliad above the rule, Odyssey below.", max_h=18*cm)
    story.append(PageBreak())

    # 4.2 formularity
    fhi = form.sort_values("repeated_line_rate", ascending=False).head(4)
    flo = form.sort_values("repeated_line_rate").head(4)
    frows = [[r.book, f"{r.repeated_line_rate:.3f}", f"{r.rep_4gram_rate:.3f}"]
             for _, r in pd.concat([fhi, flo]).iterrows()]
    story += [
        P("4.2&nbsp;&nbsp;Formularity \u2014 strong but confounded", "h2"),
        P("Formularity is a strong signal, and a cautionary one. The Odyssey is "
          "systematically more formulaic than the Iliad (repeated-line rates around "
          "0.20 versus 0.15), and within the Iliad the least formulaic books are the "
          "battle climaxes \u2014 the river-fight (21), the death of Hector (22), the "
          "funeral games (23). This almost certainly reflects <i>content</i>: the "
          "Odyssey recycles hospitality and travel type-scenes, whereas sustained "
          "battle narrative varies more. Formularity is thus a textbook case of the "
          "content confound and cannot on its own be read as evidence about "
          "authorship (Figure&nbsp;2).", "body"),
        table(["Book", "Repeated-line rate", "Repeated 4-gram rate"],
              frows, [3.0*cm, 4.2*cm, 4.5*cm], aligns=["L", "R", "R"]),
        P("Table 4. Most (top four) and least (bottom four) formulaic books.", "caption"),
    ]
    story += fig(fdir / "formularity.png",
                 "Figure 2. Repeated whole-line rate by book. The Odyssey (blue) is "
                 "uniformly more formulaic than the Iliad (orange).", max_h=8.5*cm)

    # 4.3 richness (brief negative result)
    rhi = rich.sort_values("yule_k", ascending=False).head(3)
    rlo = rich.sort_values("yule_k").head(3)
    rrows = [[r.book, f"{r.yule_k:.1f}", f"{r.mattr_w100:.3f}"]
             for _, r in pd.concat([rhi, rlo]).iterrows()]
    story += [
        P("4.3&nbsp;&nbsp;Lexical richness \u2014 a null result", "h2"),
        P("Lexical richness barely varies: MATTR sits in a narrow band (~0.85\u20130.88) "
          "for every one of the 48 books, and Yule's K spans a modest range with no "
          "dramatic outliers. This negative result is itself worth recording \u2014 "
          "vocabulary diversity is not a useful lever for this corpus and should not "
          "be leaned on when the distance measures are built.", "body"),
        table(["Book", "Yule's K", "MATTR"], rrows,
              [2.6*cm, 3.0*cm, 3.0*cm], aligns=["L", "R", "R"]),
        P("Table 5. Richness extremes (three highest, three lowest Yule's K); the "
          "spread is small.", "caption"),
    ]
    story.append(PageBreak())

    # 4.4 within-book (new)
    top = het.sort_values("max_centroid_dist", ascending=False).head(6)
    wrows = [[r.book, str(int(r.n_lines)), str(int(r.n_changepoints)),
              (r.changepoint_lines if isinstance(r.changepoint_lines, str) else "\u2014"),
              f"{r.max_centroid_dist:.3f}"] for _, r in top.iterrows()]
    story += [
        P("4.4&nbsp;&nbsp;Within-book seams \u2014 the strongest result", "h2"),
        P("Turning the lens inside individual books produces the clearest finding so "
          "far. Ranking all 48 books by the most stylistically divergent stretch each "
          "contains puts <b>Iliad&nbsp;18 and Iliad&nbsp;2 at the top</b> "
          "(Table&nbsp;6, Figure&nbsp;5) \u2014 precisely the two most famous embedded "
          "set-pieces of the poem: the <b>Shield of Achilles</b> and the "
          "<b>Catalogue of Ships</b>. In Iliad&nbsp;18 the change-point search places "
          "a boundary at line 481, where the Shield ekphrasis begins (18.478), and the "
          "self-distance heatmap shows a clean divergent block for the description "
          "(Figure&nbsp;3). In Iliad&nbsp;2 the Catalogue is bracketed at roughly lines "
          "481 and 631 and stands out as a self-similar block distinct from the "
          "surrounding narrative (Figure&nbsp;4). The method, in other words, "
          "independently rediscovers passages that philology has always flagged as "
          "stylistically distinct.", "body"),
        P("The interpretation must stay disciplined. These stretches light up largely "
          "because a descriptive ekphrasis and a formulaic list differ in "
          "<i>register</i> and <i>content</i> from ordinary narrative \u2014 not "
          "because a different poet demonstrably wrote them. A within-book seam is a "
          "candidate for scrutiny, not a verdict; separating \u201cdifferent "
          "register\u201d from \u201cdifferent hand\u201d is exactly what the "
          "narrator/character-speech split in the plan is for.", "body"),
        table(["Book", "Lines", "Change pts", "Change-point lines", "Max divergence"],
              wrows, [2.0*cm, 1.6*cm, 2.0*cm, 4.4*cm, 3.0*cm],
              aligns=["L", "R", "R", "L", "R"]),
        P("Table 6. The six books containing the most stylistically divergent internal "
          "stretch (max window-to-centroid distance).", "caption"),
    ]
    story += fig(fdir / "within_IL18.png",
                 "Figure 3. Within-book stylometry, Iliad 18. The rolling curve climbs "
                 "through the Shield of Achilles (gold); the change point (dashed) and "
                 "the heatmap block mark its onset at line ~481.", max_h=13*cm)
    story += fig(fdir / "within_IL02.png",
                 "Figure 4. Within-book stylometry, Iliad 2. The Catalogue of Ships "
                 "(gold) forms a self-similar block distinct from the surrounding "
                 "narrative.", max_h=13*cm)
    story += fig(fdir / "within_book_heterogeneity.png",
                 "Figure 5. Internal heterogeneity of all 48 books. Iliad 18 (Shield) "
                 "and Iliad 2 (Catalogue) are clear outliers.", max_h=8*cm)

    # ── 5. Interpretation ──
    story += [
        P("5&nbsp;&nbsp;Interim interpretation", "h1"),
        P("Of the book-level signals, richness is a null and formularity is entangled "
          "with content; the particle profile is the one clean stylistic signal, and "
          "it aligns with the traditional division into two poems \u2014 though it must "
          "now be re-checked once both poems are drawn from a single edition. The "
          "within-book analysis is the most compelling result: a content-light seam "
          "detector recovers the Shield and the Catalogue without being told to look "
          "for them, which validates the method even as it underscores the confound "
          "(those passages differ in register, not provably in authorship).", "body"),
        P("Nothing here is yet an answer to one-versus-many. That requires inter-book "
          "<b>distance measures and clustering</b>, <b>calibration</b> against known "
          "single-author hexameter, and <b>orthogonal signals</b> (character, metre, "
          "syntax) to corroborate any boundary the word level suggests \u2014 all set "
          "out below.", "body"),
    ]

    # ── 6. Limitations ──
    story += [
        P("6&nbsp;&nbsp;Limitations", "h1"),
        P("<b>Mixed editions.</b> The Iliad (Monro\u2013Allen) and Odyssey (Murray) "
          "come from different editors, so cross-poem differences \u2014 including the "
          "particle and formularity results \u2014 may be partly editorial rather than "
          "authorial. Within-book results are unaffected. Resolving this is the plan's "
          "first task.", "body"),
        P("<b>Normalisation and elision.</b> Because elision marks are stripped, an "
          "elided \u03b4\u1fbd is not currently counted as \u03b4\u03ad; editors elide "
          "at different rates, which can distort particle counts across poems. The "
          "normalisation needs an elision-aware pass.", "body"),
        P("<b>Surface forms only.</b> No lemmatisation, part-of-speech, syntax, or "
          "metre yet, so morphology and the orthogonal metrical signal are unexploited. "
          "A few particles are also ambiguous once accents are stripped "
          "(\u1f22, \u03c9\u03c2, \u03bf\u03c5).", "body"),
        P("<b>Change-point sensitivity.</b> The within-book detector's output depends "
          "on a penalty parameter; boundaries should be read together with the "
          "continuous curve and heatmap, not as hard claims.", "body"),
        P("<b>No calibration.</b> Between-poem and within-book distances are not yet "
          "benchmarked against genuine between-author distances, so their magnitude "
          "cannot yet be interpreted \u2014 the deepest gap, addressed next.", "body"),
    ]

    # ── 7. Plan ──
    story += [
        P("7&nbsp;&nbsp;Plan for continuation", "h1"),
        P("The next phases are organised to close the limitations above, roughly in "
          "order of leverage.", "body"),
        P("<b>1. Matched edition (fixes \u00a76 \u201cmixed editions\u201d).</b> Re-source "
          "both poems from a single editorial tradition \u2014 ideally the Oxford "
          "Classical Text throughout (Allen edited both the Iliad, with Monro, and the "
          "OCT Odyssey), sourcing the OCT Odyssey from the OpenGreekAndLatin / "
          "First1KGreek corpus and verifying the editor field matches. Re-run the whole "
          "pipeline; only then are the cross-poem particle and formularity results "
          "trustworthy.", "num"),
        P("<b>2. Elision-aware normalisation + lemmatisation (fixes \u201celision\u201d "
          "and \u201csurface forms\u201d).</b> Add an elision-aware token pass so "
          "\u03b4\u1fbd counts as \u03b4\u03ad, then lemmatise and POS-tag with CLTK / "
          "Stanza to reduce sparsity and unlock syntactic features.", "num"),
        P("<b>3. Distance matrices and clustering (the payoff).</b> Build three "
          "independent book\u00d7book distance matrices \u2014 Burrows's Delta on the "
          "stored most-frequent-word matrix, character n-grams, and a character-level "
          "cross-entropy grid \u2014 with hierarchical and bootstrap clustering. This "
          "is where the exploratory work becomes an actual test of whether the "
          "material clusters as one voice or several.", "num"),
        P("<b>4. Calibration with an anchor corpus (fixes \u201cno calibration\u201d).</b> "
          "Run known single-author Homerising hexameter (Apollonius' Argonautica, "
          "Quintus' Posthomerica) and other archaic verse (Hesiod, the Homeric Hymns) "
          "through the same pipeline, so within-Homer distances can be judged against "
          "real between-author distances. Include sanity checks: a single-author work "
          "must cluster as one; a deliberately stitched pseudo-corpus must show seams.", "num"),
        P("<b>5. Orthogonal signals \u2014 metre and syntax.</b> Add automatic "
          "hexameter scansion (foot shapes, caesura, bridge violations) and "
          "dependency-syntax features from the Perseus treebank (dependency distance, "
          "parataxis vs. hypotaxis, enjambement). Metre in particular is a "
          "content-free channel independent of the word level, so agreement across "
          "them is strong evidence.", "num"),
        P("<b>6. Narrator vs. character-speech split (the key disentangler).</b> Tag "
          "each line as narration or quoted speech and re-run every feature on each "
          "stratum. This tests whether the <i>narrating</i> voice is uniform across "
          "books and whether within-book seams such as the Shield and Catalogue are "
          "merely register shifts \u2014 directly addressing the confound that qualifies "
          "the current results.", "num"),
        P("<b>7. Statistical endpoint.</b> Fold the descriptive distances into a "
          "hierarchical (Dirichlet-process) mixture that returns a posterior over the "
          "number of distinguishable voices with honest uncertainty, plus a "
          "sensitivity analysis over modelling assumptions. The deliverable is not a "
          "headcount but a map of which conclusions the data support and which it "
          "cannot decide.", "num"),
    ]

    # ── References ──
    refs = [
        "Bakker, E. J. (1997). <i>Poetry in Speech: Orality and Homeric Discourse</i>. Cornell University Press.",
        "Burrows, J. (2002). \u2018Delta\u2019: a measure of stylistic difference and a guide to likely authorship. <i>Literary and Linguistic Computing</i>, 17(3), 267\u2013287.",
        "Covington, M. A., &amp; McFall, J. D. (2010). Cutting the Gordian knot: the moving-average type\u2013token ratio (MATTR). <i>Journal of Quantitative Linguistics</i>, 17(2), 94\u2013100.",
        "Dunning, T. (1993). Accurate methods for the statistics of surprise and coincidence. <i>Computational Linguistics</i>, 19(1), 61\u201374.",
        "Lord, A. B. (1960). <i>The Singer of Tales</i>. Harvard University Press.",
        "Monro, D. B., &amp; Allen, T. W. (eds.) (1920). <i>Homeri Opera</i> I\u2013II (Oxford Classical Texts, 3rd ed.). Clarendon Press. [Iliad; via Perseus.]",
        "Murray, A. T. (ed./trans.) (1919). <i>Homer: The Odyssey</i>. Heinemann. [Greek text via Perseus.]",
        "Nagy, G. (1996). <i>Poetry as Performance: Homer and Beyond</i>. Cambridge University Press.",
        "Parry, M. (1971). <i>The Making of Homeric Verse: The Collected Papers of Milman Parry</i> (A. Parry, ed.). Clarendon Press.",
        "Truong, C., Oudre, L., &amp; Vayatis, N. (2020). Selective review of offline change point detection methods. <i>Signal Processing</i>, 167, 107299. [PELT / ruptures.]",
        "Yule, G. U. (1944). <i>The Statistical Study of Literary Vocabulary</i>. Cambridge University Press.",
    ]
    story += [P("References", "h1")] + [P(r, "ref") for r in refs]
    return story


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).resolve().parent
    out_dir = root / "outputs" / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "homer_stylo_report.pdf"
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm, topMargin=1.8*cm, bottomMargin=1.8*cm,
        title="Is Homer One Voice? — Project report (Phases 0–2b)", author="Fada",
    )
    doc.build(build(cfg, root))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
