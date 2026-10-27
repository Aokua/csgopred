# Reproducibility

This directory contains everything needed to reproduce the paper's experiments. All
randomised procedures use `seed = 42`, and evaluation is strictly walk-forward.

## 1. Data snapshot

| Item | Value |
|---|---|
| Database | `data/csgopred.db` |
| Snapshot hash | `database.sha256` in `repro/snapshot.json` |
| Model | `data/model.pkl`, see `model.sha256` |
| Row counts | matches / maps / news / news_events in `counts` |

Generate or verify the snapshot:

```bash
python repro/make_snapshot.py
```

Compare the sha256 values in `snapshot.json` with the local files to confirm the data
matches what the paper used.

## 2. Environment

```bash
python -c "import sys,numpy,sklearn;print(sys.version);print(numpy.__version__, sklearn.__version__)"
```

Exact versions are recorded in `snapshot.json` under `python`, `numpy` and `sklearn`.

## 3. Reproducing training

```bash
export PYTHONPATH=src
python repro/train.py
```

Outputs:

- `repro/results.json` — log loss for two feature sets on three time splits
- `data/model_repro.pkl` — model retrained on the recommended 25-feature set

The numbers in Tables 5-1 and 5-4 of the paper can be checked directly against
`results.json` and the JSON files in `paper/data/`.

## 4. Reproducing experiments

| Script | Output | Paper section |
|---|---|---|
| `experiments/decompose.py` | `paper/data/decomp.json` | 5.5 (Figures 5-1, 5-2) |
| `experiments/feature_sets.py` | `paper/data/feature_sets.json` | Table 5-4 |
| `experiments/format_revenge.py` | `paper/data/format_revenge.json` | 5.7.1 |
| `experiments/revenge_groups.py` | console output | 5.7.2 |
| `experiments/rematch_calibration.py` | console output | 5.7.2 |
| `experiments/make_figures_en.py` | `paper/images_en/*.png` | Figures 3-1, 5-1 to 5-4, 6-1 |

Run them in order:

```bash
export PYTHONPATH=src
python experiments/decompose.py
python experiments/feature_sets.py
python experiments/format_revenge.py
python experiments/revenge_groups.py
python experiments/rematch_calibration.py
python experiments/make_figures_en.py
```

## 5. Configuration

`repro/config.json` collects:

- hyperparameters (C = 0.5, lbfgs, recency half-life 1,500 days, calibration fraction 0.15)
- Elo and map-Elo parameters
- the three feature lists (base20 / base+map / recommended25)
- the time split dates

## 6. Known limitations

- The database is a research snapshot and does not include raw web pages or crawl logs.
- News events were extracted by an LLM; `news_events` keeps the supporting quote for
  each event so that extraction can be audited.
- Map coverage is about 17% of matches; uncovered matches receive neutral feature values.
