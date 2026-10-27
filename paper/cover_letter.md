# Cover letter

**To:** Editor-in-Chief, *Journal of Statistics and Data Science in Sports*

**Re:** Submission of "Design and Implementation of a Win-Probability Prediction and Calibration System for CS2 Professional Matches"

Dear Editor,

I am pleased to submit the manuscript above for consideration as an original article in
the *Journal of Statistics and Data Science in Sports*.

**What the paper does.** It presents an end-to-end, reproducible system that predicts
professional Counter-Strike 2 match outcomes from 68,523 matches (2015–2026). The model
combines team-level Elo ratings, opponent-adjusted form, head-to-head history and a
map-level Elo with a veto simulation that converts per-map strengths into series
probabilities. Under strict walk-forward evaluation the model reaches a log loss of
0.6204 (Elo baseline 0.6290), 65.4% accuracy and expected calibration error below 1.9%.

**Why it fits JSDSS.** Three aspects align with the journal's scope and principles.

1. *eSports as a laboratory.* The paper addresses a domain the journal explicitly lists
   in scope, and it does so with a large, heterogeneous public dataset rather than a
   single tournament.
2. *Reproducibility by construction.* Every feature is as-of, the evaluation is
   walk-forward, and all results can be regenerated from the accompanying repository:
   a data snapshot with SHA-256 hashes, fixed seed, training script and experiment
   scripts, plus a Quarto manuscript that draws the figures and summary tables from
   code. The repository is https://github.com/Aokua/csgopred (private at submission;
   access will be granted to reviewers and made public upon acceptance).
3. *Honest negative results.* Beyond the positive results, the paper reports a
   feature-attribution study showing that format and tier features are harmful out of
   sample, and it tests and rejects two widely held business hypotheses (explicit
   map-count modelling and an "analyst/revenge" effect after recent losses). We believe
   reporting these failures is useful to the community and consistent with the journal's
   emphasis on credibility.

**Declarations.** The manuscript is original, has not been published previously, and is
not under consideration elsewhere. The author is Zixuan Tang (Southwest Petroleum
University, Chengdu, China). There is no external funding and no competing interest. The
work uses publicly available match data only; market prices are used for evaluation and
are never model features. No betting advice is provided.

**Data and code availability.** Data are assembled from public sources (hojlund dataset,
HLTV, Kaggle, Liquipedia) and redistributed as a research snapshot together with the
code required to reproduce every table and figure.

Thank you for considering the manuscript.

Sincerely,

Zixuan Tang
Southwest Petroleum University, Chengdu, China
