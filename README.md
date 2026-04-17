# Amazon Electronics Recommendation System — Group 5

IE University · Recommendation Engines · Prof. Ignacio de Córdoba · 2025–26

## Team members
- Antonio Meneses
- Julián Consuegra
- Juan Pablo Miró-Quesada
- Tomás Roschge
- [Name 5]

## Project description

End-to-end recommendation pipeline on Amazon Electronics reviews (McAuley Lab / Stanford SNAP).
We implement and compare four approaches:

1. **Non-Personalized** — Global Mean, Item Mean, Damped Mean (Bayesian Average)
2. **Collaborative Filtering** — SVD matrix factorization via scikit-surprise
3. **Content-Based** — TF-IDF + category + brand + price feature vectors, KNN retrieval
4. **Context-Aware** — SVD on context-debiased ratings (month × day-of-week × time-period)

A fifth notebook performs a standardised comparative evaluation across all four approaches,
including prediction accuracy, ranking quality, beyond-accuracy metrics, and A/B test design.

## Dataset

| Property | Value |
|---|---|
| Source | McAuley Lab Amazon Reviews (Stanford SNAP) |
| Domain | Electronics |
| Raw reviews | 7,824,482 |
| After 5-core filtering | 2,109,869 reviews · 253,994 users · 145,199 items |
| Sparsity | 99.9943% |
| Temporal split (80/20) | Train: 1,687,895 · Test: 421,974 |

The CSV files are **not** included in the repository (too large). Download instructions below.

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd group5-rec-engines

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies (numpy<2 required for scikit-surprise)
pip install -r requirements.txt

# 4. Place data files in data/
#    amazon_electronics_filtered.csv
#    amazon_meta_filtered.csv
```

## Repo structure

```
group5-rec-engines/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                          (gitignored — place CSVs here)
├── src/
│   ├── __init__.py
│   └── utils.py                   (shared: data loading, metrics, helpers)
├── notebooks/
│   ├── 01_non_personalized.ipynb
│   ├── 02_collaborative_filtering.ipynb
│   ├── 03_content_based.ipynb
│   ├── 04_context_aware.ipynb
│   └── 05_evaluation.ipynb
├── results/                       (pkl files produced by notebooks 01–04)
└── Group_Assignment_Reccomendation_Engines_v2.ipynb  (original monolithic notebook)
```

## How to reproduce

Run the notebooks **in order**:

```
01 → 02 → 03 → 04 → 05
```

Notebooks 01–04 each save a `.pkl` file to `results/`.
Notebook 05 loads all four `.pkl` files and produces the final comparison table.

Each notebook runs independently (no live kernel state shared between them).

## Results summary

| Approach | RMSE | MAE | Precision@10 | Recall@10 | NDCG@10 | Coverage | Diversity | Serendipity |
|---|---|---|---|---|---|---|---|---|
| Non-Personalized (Damped Mean) | 1.1456 | 0.8897 | 0.0026 | 0.0089 | 0.0058 | 0.0001 | TBD | TBD |
| Collaborative Filtering (SVD) | 1.1138 | 0.8393 | 0.0001 | 0.0004 | 0.0002 | 0.0385 | TBD | TBD |
| Content-Based | 1.3364 | 0.9265 | 0.0010 | 0.0041 | 0.0028 | 0.0543 | 0.3398 | TBD |
| Context-Aware | 1.1138 | 0.8366 | 0.0002 | 0.0006 | 0.0003 | 0.0208 | TBD | TBD |

*Low absolute ranking values are expected: catalog has 134K items and test window is 6 months.*

## Submission files

| File | Description |
|---|---|
| `GROUP_5_report.md` | Full written report |
| `GROUP_5_executive_summary.md` | Executive summary (1–2 pages) |
| `GROUP_5_slides.pdf` | Presentation slides |
