# Is Homer One Voice?

### A computational-stylometric investigation of the Iliad and Odyssey

*Project report · Phases 0–3 (corpus, EDA, within-book analysis, clustering, narrator/speech split) · Fada · July 2026*

---

## Abstract

This project asks whether Homer's verse is the work of a single poetic voice or of many. The working corpus is the Greek text of both poems (27,794 verse lines, ~199k tokens) from the Perseus Digital Library. Rather than compare the two poems to each other, the emphasis is on detecting plurality of voice **within** the text. Four results anchor the report. A book-level **particle profile** and **formularity** both distinguish the poems, but the second is confounded with content and, as clustering shows, much of the apparent poem gap is an artefact of the two source editions rather than deep style. The strongest finding is internal: a **within-book seam detector independently rediscovers the Shield of Achilles and the Catalogue of Ships** as the passages most distinct from their surroundings. Finally, a **narrator vs. character-speech split** (50% of lines are direct speech) lets us strip out register: run on narration alone, the Shield seam **survives**, which is the kind of evidence a search for more than one Homer needs. All findings are exploratory; the plan sets out calibration and orthogonal signals as the next steps.

## 1  Research question and design

The “Homeric Question” is several questions at once. At the level of composition, Parry and Lord's oral-formulaic theory recast the poems as the sediment of a tradition of many singers; at the level of discourse, Bakker reads the verse as structured speech; and at the level of the text, Nagy's evolutionary model denies there was ever a single fixed original. The question this report pursues is the sharpest form of “more than one voice”: is there detectable stylistic **plurality within the text** — seams that might mark different hands — rather than how the two poems compare as wholes.

The central hazard is that most measurable differences between stretches of Homer track **content** (what a passage is about) or **register** (narration vs. speech) rather than authorial **habit**. The project is therefore built around one discipline: judge every feature by whether it measures content, register, or habit, and treat only habit as a candidate voice signal.

## 2  Corpus and source

Our source is the Greek text of both poems from the **Perseus Digital Library**, parsed from canonical TEI XML whose explicit book and line markup makes segmentation exact. A Phase-0 ingester produces a tidy line table with verbatim, diplomatic, and normalised text columns (the normalised form is lower-cased, with accents, punctuation and elision marks removed and final sigma folded); the corpus is 98% Greek.

The two poems come from different editions (the Iliad in Monro & Allen's Oxford Classical Text, the Odyssey in Murray's). Where this matters — in the cross-poem comparisons — it is flagged at the point it arises; it does not affect any within-text analysis.

| Poem | Edition (via Perseus) | Books | Lines | Tokens |
|---|---|---|---|---|
| Iliad | Monro & Allen (OCT) | 24 | 15,687 | 111,895 |
| Odyssey | Murray | 24 | 12,107 | 87,189 |
| **Total** | — | 48 | **27,794** | **199,084** |

*Table 1. Working corpus.*

## 3  Method

**Book level.** Four analyses run over the normalised tokens of each of the 48 books: a particle / function-word profile (z-scored per-mille rates of ~30 Homeric particles — content-independent, hence a habit signal); lexical richness (Yule's K, moving-average type–token ratio, hapax rate); formularity (share of a book's lines that recur elsewhere, plus repeated n-gram density); and a most-frequent-word matrix for the distance phase.

**Within-book level.** A sliding window (60 lines, step 10) moves through a single book; each window is described by its character-trigram profile. Three readings follow: the cosine distance of each window to the book's own centroid, a window×window self-distance matrix (block structure signals a seam), and change points from a penalised search (PELT).

**Between-book level.** Three distance matrices over the 48 books — Burrows's Delta (word level, robust to spelling), character-trigram cosine (sensitive to orthography), and a compression distance (NCD) — are clustered and projected to two dimensions. The narrator/speech split (§5) adds a register stratification on top of all of these.

## 4  Results

### 4.1  Particle profile

The two poems separate on function-word usage: *δέ* is markedly commoner in the Iliad, while *τοι*, *καί*, *αὐτάρ* and *ἢ* lean to the Odyssey, and *ἤτοι* is effectively Iliad-only. Because these words carry almost no subject matter, the difference is a genuine habit signal (part of it may nonetheless be editorial, given §2 — see §4.5).

| Particle | Iliad (‰) | Odyssey (‰) | Δ | Leans |
|---|---|---|---|---|
| τοι | 3.61 | 6.23 | -2.62 | Odyssey |
| και | 25.80 | 28.05 | -2.25 | Odyssey |
| η | 7.31 | 9.27 | -1.97 | Odyssey |
| αυταρ | 3.20 | 4.61 | -1.41 | Odyssey |
| περ | 3.01 | 2.40 | +0.61 | Iliad |
| ρα | 1.92 | 1.03 | +0.88 | Iliad |
| ητοι | 0.96 | 0.00 | +0.96 | Iliad |
| δε | 24.26 | 19.56 | +4.70 | Iliad |

*Table 3. Largest Iliad–Odyssey divergences in particle frequency (per mille).*

![Figure 1. Particle / function-word profile by book (per-mille, z-scored).](figures/particle_heatmap.png)

*Figure 1. Particle / function-word profile by book (per-mille, z-scored).*

### 4.2  Formularity — strong but confounded

The Odyssey is systematically more formulaic than the Iliad (repeated-line rates around 0.20 vs. 0.15), and within the Iliad the least formulaic books are the battle climaxes. This almost certainly reflects content — the Odyssey recycles hospitality and travel type-scenes — so formularity cannot on its own be read as evidence about authorship.

| Book | Repeated-line rate | Repeated 4-gram rate |
|---|---|---|
| OD17 | 0.271 | 0.257 |
| IL08 | 0.257 | 0.275 |
| OD01 | 0.243 | 0.255 |
| OD04 | 0.229 | 0.231 |
| IL21 | 0.051 | 0.094 |
| IL23 | 0.094 | 0.105 |
| OD06 | 0.094 | 0.123 |
| IL22 | 0.101 | 0.124 |

*Table 4. Most (top four) and least (bottom four) formulaic books.*

![Figure 2. Repeated whole-line rate by book.](figures/formularity.png)

*Figure 2. Repeated whole-line rate by book.*

### 4.3  Lexical richness — a null result

Lexical richness barely varies: MATTR sits in a narrow band (~0.85–0.88) for all 48 books and Yule's K spans a modest range with no dramatic outliers. A useful negative: vocabulary diversity is not a lever for this corpus.

| Book | Yule's K | MATTR |
|---|---|---|
| OD06 | 43.0 | 0.855 |
| IL03 | 42.4 | 0.864 |
| IL23 | 42.3 | 0.861 |
| OD11 | 32.3 | 0.873 |
| OD01 | 33.9 | 0.879 |
| IL05 | 34.1 | 0.872 |

*Table 5. Richness extremes (three highest, three lowest Yule's K).*

### 4.4  Within-book seams — the strongest result

Ranking all 48 books by the most stylistically divergent stretch each contains puts **Iliad 18 and Iliad 2 at the top** — precisely the two most famous embedded set-pieces: the **Shield of Achilles** and the **Catalogue of Ships**. In Iliad 18 the change-point search places a boundary at line 481, where the Shield ekphrasis begins (18.478); in Iliad 2 the Catalogue forms a self-similar block distinct from the surrounding narrative. The method rediscovers, unprompted, passages philology has always flagged.

| Book | Lines | Change pts | Change-point lines | Max divergence |
|---|---|---|---|---|
| IL18 | 617 | 1 | 481 | 0.137 |
| IL02 | 877 | 2 | 481;631 | 0.110 |
| IL04 | 544 | 2 | 231;431 | 0.101 |
| OD09 | 566 | 1 | 281 | 0.096 |
| OD02 | 434 | 0 | — | 0.096 |
| OD19 | 604 | 2 | 231;381 | 0.095 |

*Table 6. The six books containing the most divergent internal stretch.*

![Figure 3. Within-book stylometry, Iliad 18: the curve climbs through the Shield (gold); the change point and heatmap block mark its onset.](figures/within_IL18.png)

*Figure 3. Within-book stylometry, Iliad 18: the curve climbs through the Shield (gold); the change point and heatmap block mark its onset.*

![Figure 4. Within-book stylometry, Iliad 2: the Catalogue of Ships as a distinct block.](figures/within_IL02.png)

*Figure 4. Within-book stylometry, Iliad 2: the Catalogue of Ships as a distinct block.*

![Figure 5. Internal heterogeneity of all 48 books; Iliad 18 and 2 are outliers.](figures/within_book_heterogeneity.png)

*Figure 5. Internal heterogeneity of all 48 books; Iliad 18 and 2 are outliers.*

These stretches differ in **register and content** (a descriptive ekphrasis, a formulaic list) from ordinary narrative, so a within-book seam is a candidate for scrutiny, not a verdict — which is exactly what motivates the register split in §5.

### 4.5  Between-book distances and clustering

Clustering the 48 books is instructive because the measures disagree. On **character trigrams the two poems separate perfectly** (100% pure by poem). On **Burrows's Delta they barely separate** — an overlapping cloud with only a weak tendency, and several books crossing over. This disagreement is the fingerprint of the two editions (§2): character n-grams track orthographic convention, which is uniform within an edition, so two editions split cleanly whether or not their authors differ. Delta, resting on function words, is robust to spelling and is the trustworthy cross-poem signal — and it says the poems are only **modestly** distinguishable (Delta–character correlation r = 0.74; the compression distance is a weak, largely independent signal, r = 0.07, and is discounted). The clean character split should not be read as an authorial boundary.

| Distance measure | Poem separation (2-cluster purity) |
|---|---|
| Character trigrams | 100% (perfect — but see text) |
| Burrows's Delta (words) | 52% (poems are not the top split) |
| Compression (NCD) | 52% (weak, discounted) |

*Table 7. How cleanly each measure splits the poems; the character split is an edition effect.*

![Figure 6. Books in stylistic space (MDS on Burrows's Delta); the poems overlap.](figures/mds_delta.png)

*Figure 6. Books in stylistic space (MDS on Burrows's Delta); the poems overlap.*

![Figure 7. Character-trigram clustering: a clean two-block split — the signature of two editions.](figures/clustermap_charcos.png)

*Figure 7. Character-trigram clustering: a clean two-block split — the signature of two editions.*

![Figure 8. Word-level (Delta) clustering: only a partial poem tendency, with crossovers.](figures/clustermap_delta.png)

*Figure 8. Word-level (Delta) clustering: only a partial poem tendency, with crossovers.*

## 5  The narrator vs. character-speech split

### 5.1  How the tagger was built

Homer alternates between the poet's narration and quoted character speech, and that alternation is a register shift that would masquerade as a change of hand. Tagging the two strata is therefore the key disentangler. The starting point was a check of the raw text:

> No quotation marks are present, but the speech-boundary formulae are all there once the normalised final sigma is handled correctly — προσέφη (234), ἀμειβόμενος (208), ὣς φάτο (197), προσηύδα (176), and so on. That is the classic way to detect Homeric speech: an introduction formula opens a speech, and a resumption formula (ὣς φάτο / ὣς εἰπών) closes it.

The tagger built on that idea then needed three corrections, each prompted by inspecting its output:

1. **Precision over recall.** An initial, generous cue list (including φωνήσας and ἀγορεύω) let speech bleed into narration, because “ὣς ἄρα φωνήσας” resumptions were also being read as openers. The cues were pared back to the reliable reply/address frames, which pair with a resumption and so yield clean spans.

2. **Reply-frames close *and* open.** A dialogue spot-check (Iliad 1) showed that in rapid exchange a speech is often not closed by ὣς φάτο but runs straight into the next frame (τὸν δ᾽ ἠμείβετ᾽ …); the frame line itself was being absorbed into the speech. The fix: a reply frame both closes the current speech and opens the next — it is narration either way.

3. **Augmented forms.** The augmented reply verb ἠμείβετο (normalised “ημειβετ”) was missed by a cue written for the α- forms; the cue was broadened to catch both.

With these in place the split lands at a share of speech that matches the scholarly estimate for Homer, and dialogue segments cleanly (the frame lines come out as narration, the speeches between them as speech).

### 5.2  Validation and the register contrast

The corpus divides into **13,934 narration lines and 13,860 speech lines (49.9% speech; 50.4% by tokens)** — right on the received estimate that roughly half of Homer is direct speech. The contrast between the strata is exactly what it should be (Figure 9): speech is marked by vocatives (γέρον, Ὀδυσσεῦ), second-person address, an imperative (κέκλυτε “hear!”), possessives (ἐμόν, σόν) and the dual νῶι “we two”; narration is marked by the speech-frame verbs and third-person report.

![Figure 9. What distinguishes character speech from narration (most discriminating frequent words).](figures/voice_markers.png)

*Figure 9. What distinguishes character speech from narration (most discriminating frequent words).*

### 5.3  Within-narration seams — probing for more than one Homer

The split's payoff is that a seam can now be tested on the narrating voice alone. Re-running the Iliad 18 seam detector on **narration only** (speech removed) is the decisive check for the Shield of Achilles: on the full text the Shield boundary sits at line 481 (max divergence 0.137); on narration only, at matched resolution, a change point re-appears inside the ekphrasis (line 504) and the maximum divergence actually **rises to 0.152**. The Shield seam therefore **survives the removal of speech** — it is a genuine shift in the narrating voice, not an artefact of a character starting to talk. That is precisely the sort of within-text discontinuity a search for more than one Homer is looking for (a candidate, not yet a verdict).

![Figure 10. Iliad 18, narration only: the Shield block persists and a change point re-appears inside the ekphrasis after speech is removed.](figures/within_IL18_narration.png)

*Figure 10. Iliad 18, narration only: the Shield block persists and a change point re-appears inside the ekphrasis after speech is removed.*

### 5.4  Limitation of the split

The tagger is a formula-based heuristic that favours **precision over recall**. Speeches opened by a plain lexical verb rather than a reply frame — prayers (ἠρᾶτο), commands (μῦθον ἔτελλε), the first plea of a scene (λίσσετο) — are missed and stay labelled narration (Chryses' public plea in Iliad 1 is one such case). The narration stratum therefore carries a little residual scene-opening speech, and the reported speech share is a **lower bound**. This is a metric limitation to keep in view when reading the stratified results, and a candidate for a trained tagger in future work.

## 6  Interim interpretation

The cross-poem signals are the weakest ground: richness is a null, formularity is content-bound, and clustering shows the crisp character-level poem split is largely an edition artefact, with only a modest difference at the word level. The centre of gravity is inside the text. The within-book seam detector recovers the Shield and the Catalogue unprompted, and — the key new result — the Shield seam **persists once speech is stripped out**, so it reflects the narrating voice rather than register. This is the most promising lead for the “more than one Homer” question, though it remains a lead: a distinctive stretch is not yet a demonstrated second hand.

## 7  Limitations

- **Mixed editions.** The Iliad (Monro–Allen) and Odyssey (Murray) come from different editors, so cross-poem differences may be partly editorial; §4.5 shows this directly. Within-text results are unaffected.
- **Speech-tagger recall.** The narrator/speech split is precision-favouring and misses lexically-opened speeches, so the narration stratum holds some residual speech and the speech share is a lower bound (§5.4).
- **Change-point sensitivity.** The seam detector's output depends on a penalty and on window resolution; boundaries should be read together with the continuous curve and heatmap, not as hard claims.
- **Surface forms only.** No lemmatisation, part-of-speech, syntax, or metre yet, so morphology and the orthogonal metrical signal are unexploited; a few particles are ambiguous once accents are stripped.
- **No calibration.** Within-text distances are not yet benchmarked against genuine between-author distances, so their magnitude cannot yet be interpreted — the deepest gap.

## 8  Plan for continuation

1. **Systematic within-narration seam scan.** Extend the Iliad-18 probe to every book on the narration stratum, cataloguing seams that survive speech removal — the direct hunt for internal plurality.
2. **Calibration with an anchor corpus.** Run known single-author hexameter (Apollonius, Quintus) and other archaic verse (Hesiod, the Hymns) through the same pipeline, so within-text seam magnitudes can be judged against real between-author differences; include sanity checks (a single-author work must look seamless; a stitched pseudo-corpus must show a seam).
3. **Orthogonal signals — metre and syntax.** Add hexameter scansion (foot shapes, caesura, bridges) and dependency-syntax features; metre is a content-free channel, so agreement with the word level is strong evidence.
4. **Elision-aware normalisation + lemmatisation** to remove orthographic noise and reduce sparsity.
5. **A single edition for both poems**, so any residual cross-poem work is free of the orthographic confound §4.5 exposes.
6. **Statistical endpoint.** A hierarchical (Dirichlet-process) mixture returning a posterior over the number of distinguishable voices, with a sensitivity analysis — a map of what the data support, not a headcount.

## References

1. Bakker, E. J. (1997). *Poetry in Speech: Orality and Homeric Discourse*. Cornell University Press.
2. Burrows, J. (2002). 'Delta': a measure of stylistic difference and a guide to likely authorship. *Literary and Linguistic Computing*, 17(3), 267–287.
3. Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot: the moving-average type–token ratio (MATTR). *Journal of Quantitative Linguistics*, 17(2), 94–100.
4. Dunning, T. (1993). Accurate methods for the statistics of surprise and coincidence. *Computational Linguistics*, 19(1), 61–74.
5. Kešelj, V., Peng, F., Cercone, N., & Thomas, C. (2003). N-gram-based author profiles for authorship attribution. *PACLING*, 255–264.
6. Li, M., Chen, X., Li, X., Ma, B., & Vitányi, P. (2004). The similarity metric. *IEEE Trans. Information Theory*, 50(12), 3250–3264.
7. Lord, A. B. (1960). *The Singer of Tales*. Harvard University Press.
8. Nagy, G. (1996). *Poetry as Performance: Homer and Beyond*. Cambridge University Press.
9. Parry, M. (1971). *The Making of Homeric Verse* (A. Parry, ed.). Clarendon Press.
10. Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of offline change point detection methods. *Signal Processing*, 167, 107299.
