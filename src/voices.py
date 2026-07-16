"""
voices.py — the statistical endpoint: how many voices?

The dispersion tests say the poems are over-varied; the low cluster silhouettes
say that variation is *diffuse*, not split into discrete groups. This turns the
"diffuse vs discrete" question into a number: for each work, how many stylistic
components does the data actually support?

Two independent estimators over the 400-line-chunk feature space (Burrows z-scored,
PCA-reduced):
  • Gaussian-mixture model, number of components chosen by BIC;
  • Dirichlet-process (Bayesian) mixture, counting effectively-occupied components.

If Homer's counts are no higher than the single-author epics' (all ≈ 1), then even
Homer shows no discrete voices — the strong-sense "no". If Homer supports more
components than a known single author, that is quantitative evidence for
separable hands.

Run:  python src/voices.py --config config.yaml
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

from bootstrap import load_works, word_features, metre_features
from calibrate import balanced_zscore, SINGLE_AUTHOR

ORDER = ["Iliad", "Odyssey", "Apollonius", "Quintus", "Hesiod"]


def estimate_k(X, kmax):
    """BIC-optimal component count and DP effective component count for points X."""
    n = len(X)
    kmax = max(1, min(kmax, n // 5))
    best_k, best_bic = 1, np.inf
    for k in range(1, kmax + 1):
        try:
            gm = GaussianMixture(k, covariance_type="full", reg_covar=1e-2,
                                 random_state=0, n_init=3).fit(X)
            b = gm.bic(X)
            if b < best_bic:
                best_bic, best_k = b, k
        except Exception:
            pass
    dp_k = np.nan
    if n >= 6:
        try:
            K = max(2, kmax)
            bg = BayesianGaussianMixture(
                n_components=K, covariance_type="full", reg_covar=1e-2,
                weight_concentration_prior_type="dirichlet_process",
                random_state=0, n_init=2, max_iter=800).fit(X)
            dp_k = int((bg.weights_ > 1.0 / (2 * K)).sum())
        except Exception:
            pass
    return best_k, dp_k


def channel_matrix(works, which):
    if which == "word":
        return word_features(works, 400, 150)
    return metre_features(works, 400)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--kmax", type=int, default=6)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).resolve().parent
    tables = root / cfg["paths"]["tables_dir"]
    norm_cfg = cfg["normalize"]["normalized"]
    works = load_works(cfg, root, norm_cfg)

    warnings.filterwarnings("ignore")
    rows = []
    for which in ["word", "metre"]:
        F, owner = channel_matrix(works, which)
        Z = balanced_zscore(F, owner, ORDER)
        ncomp = min(10, Z.shape[0] - 1, Z.shape[1])
        P = PCA(n_components=ncomp, random_state=0).fit_transform(Z)
        for name in ORDER:
            X = P[owner == name]
            bic_k, dp_k = estimate_k(X, args.kmax)
            rows.append({"channel": which, "work": name,
                         "single_author": name in SINGLE_AUTHOR,
                         "n_chunks": len(X), "bic_k": bic_k, "dp_effective_k": dp_k})
    out = pd.DataFrame(rows)
    out.to_csv(tables / "voices_summary.csv", index=False)

    print("=== HOW MANY VOICES? (effective stylistic components per work) ===\n")
    for which in ["word", "metre"]:
        sub = out[out.channel == which]
        print(f"[{which} channel]")
        print(sub[["work", "single_author", "n_chunks", "bic_k", "dp_effective_k"]]
              .to_string(index=False))
        homer = sub[sub.work.isin(["Iliad", "Odyssey"])]
        cal = sub[sub.work.isin(["Apollonius", "Quintus"])]
        print(f"  Homer BIC components: {sorted(homer.bic_k)} ; "
              f"single-epic BIC components: {sorted(cal.bic_k)}")
        verdict = ("no more than the single-author epics — the plurality is DIFFUSE, "
                   "not discrete") if homer.bic_k.max() <= cal.bic_k.max() else \
                  ("MORE than a single-author epic — possible discrete structure")
        print(f"  → Homer supports {verdict}.\n")
    print(f"Wrote {tables/'voices_summary.csv'}")


if __name__ == "__main__":
    main()
