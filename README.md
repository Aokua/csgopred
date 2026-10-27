# csgopred — CS2 win-probability and calibration model

Given two teams and a match format, the system returns a calibrated win probability,
a 95% interval, fair and reference prices, a simulated map pool with per-map
probabilities, and the main drivers. It also reports a suggested position size when
capital and market prices are supplied.

**Status:** research prototype. Evaluation is walk-forward only; market prices are not
used as features.

## Results

| Model | Features | 2025-08 | 2026-09 | Mean |
|---|---|---:|---:|---:|
| Elo baseline | 1 | 0.62999 | 0.62797 | 0.6290 |
| Logistic regression | 20 | 0.62235 | 0.62163 | 0.6220 |
| LR + map features | 27 | 0.62155 | 0.62057 | 0.6211 |
| Recommended set | 25 | 0.62076 | 0.62011 | 0.6204 |
| Gradient boosting | 27 | 0.62786 | 0.62766 | 0.6278 |

Test accuracy is 65.1% and 65.4%; expected calibration error is 0.85% and 1.89%.

## Data

| Item | Value |
|---|---|
| Matches | 68,523 (after deduplication; 69,509 raw) |
| Period | 2015-11 – 2026-10 |
| Teams | 4,564 |
| Maps | 56,730 (14,449 since 2023) |
| News articles / extracted events | 1,097 / 273 |

Sources: the hojlund dataset, an HLTV scrape, a Kaggle map-level dataset, and Liquipedia.
Team names are normalized with a four-stage matcher; BO1/BO5 labels are repaired from
the number of maps actually played.

**Data availability.** The SQLite snapshot (`data/csgopred.db`, 60 MB) and the raw
player-statistics CSV (`data/raw/kaggle/players.csv`, 129 MB) are distributed with this
repository via Git LFS, so the pipeline can be rerun end to end. Their SHA-256 hashes are
recorded in `repro/snapshot.json`.

## Quick start

```bash
export PYTHONPATH=src
pip install -r requirements.txt

python -m csgopred stats                     # dataset summary
python -m csgopred predict -a Vitality -b Spirit --format bo3
python -m csgopred serve --port 8080         # dashboard at / (zh) and /en
```

Open <http://127.0.0.1:8080> for the Chinese dashboard, <http://127.0.0.1:8080/en> for English.

## Model

- **Rating:** Elo with margin multiplier and inactivity decay; per-map Elo with
  shrinkage toward the team rating when fewer than five maps are observed.
- **Veto:** heuristic ban/pick by relative map strength, then dynamic programming over
  the remaining maps to obtain the series probability.
- **Features:** 20 base features (ratings, form, schedule, head-to-head, format/tier),
  plus map features and lineup features; the recommended set drops format/tier.
- **Model:** L2 logistic regression with recency weights (half-life 1,500 days),
  Platt-style calibration on a held-out fraction.
- **Decision layer:** theoretical price, market friction, fractional Kelly sizing with
  a 10% cap, and Monte Carlo drawdown/ruin diagnostics.

### Key finding

Format and tier features (`is_bo1`, `is_bo5`, `tier_major`, `tier_top`, `lan`, `year`)
are **harmful** out of sample: removing them improves log loss by 0.0014 on average
across three time splits (0.62199 → 0.62044 on the two primary splits). The evaluation
of two further hypotheses is negative; see the paper, Section 5.7.

## Reproducibility

```bash
python repro/make_snapshot.py   # hashes of data/model and environment versions
python repro/train.py           # retrains and writes repro/results.json
```

Experiment scripts live in `experiments/`:

| Script | Output |
|---|---|
| `decompose.py` | nested and leave-one-out attribution |
| `feature_sets.py` | feature-set comparison across three splits |
| `format_revenge.py` | map-count and rematch experiments |
| `revenge_groups.py` | rematch statistics by gap and favourite/underdog |
| `rematch_calibration.py` | Elo over-confidence check |
| `make_figures_en.py` | all English figures |

Data snapshot hashes and environment versions are recorded in `repro/snapshot.json`.

## Repository layout

```
src/csgopred/
  ingest/      HLTV, 5E, Liquipedia, Kaggle, browser automation
  features/    Elo, map Elo, pipeline, news features
  models/      logistic/HGB, calibration, pricing, sizing
  eval/        walk-forward splits, metrics, market evaluation
  serve/       CLI, HTTP API, static dashboard (zh + en)
experiments/   reproducible experiment scripts
repro/         snapshot, training and config for reproduction
paper/         manuscript (Chinese and English), figures, Quarto source
tests/         56 tests
```

## Paper

- Chinese manuscript: `paper/CS2胜率预测系统技术论文_v3.docx`
- English manuscript: `paper/paper_en_final.docx` / `.pdf`
- Quarto source for the reproducibility stage: `paper/manuscript.qmd`

## License and scope

Code is released under the **MIT License**; documentation, figures and the data snapshot
are released under **CC BY 4.0**. Third-party source data (HLTV, Liquipedia, Kaggle, the
hojlund dataset) remains subject to its own terms; see `LICENSE` for details.

Research use only. Scraping respects target-site terms and rate limits. Market prices
are used for evaluation only; nothing here is betting advice.
