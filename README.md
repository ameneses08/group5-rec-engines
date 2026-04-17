# Amazon Electronics Recommendation System — Group 5

IE University · Recommendation Engines · Prof. Ignacio de Córdoba · 2025–26

## Team members
- Antonio Meneses
- Julián Consuegra
- Juan Pablo Miró-Quesada
- Tomás Roschge

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

The CSV files are **not** included in the repository (too large for git).

## Setup

> **Do steps 1–4 before running any notebook.**

```bash
# 1. Clone the repo
git clone https://github.com/ameneses08/group5-rec-engines.git
cd group5-rec-engines

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies (numpy<2 required for scikit-surprise)
pip install -r requirements.txt

# 4. Add the data files  ← required before running any notebook
mkdir data
# Then place both CSV files in data/:
#   data/amazon_electronics_filtered.csv
#   data/amazon_meta_filtered.csv
#
# Team members:   download from the shared Google Drive (Group_Rec_Engines_Project)
# Professor/grader: the files are included in the submission zip
```

## Running on Google Colab

**Step 1: Setup (run once in a new Colab notebook)**

```python
# Clone the repo
!git clone https://github.com/ameneses08/group5-rec-engines.git
%cd group5-rec-engines

# Install dependencies (will downgrade numpy — this is expected)
!pip install -q -r requirements.txt
```

After this cell finishes, go to **Runtime → Restart runtime**. This is required because scikit-surprise needs numpy<2.

**Step 2: Load data from Google Drive**

After the runtime restarts, run this in a new cell:

```python
# Navigate back to the repo — required after every runtime restart
%cd /content/group5-rec-engines

from google.colab import drive
drive.mount('/content/drive')

import shutil, os
os.makedirs('data', exist_ok=True)
shutil.copy('/content/drive/MyDrive/Group_Rec_Engines_Project/amazon_electronics_filtered.csv', 'data/')
shutil.copy('/content/drive/MyDrive/Group_Rec_Engines_Project/amazon_meta_filtered.csv', 'data/')
print("Data files copied successfully!")
```

**Step 3: Run the notebooks**

After the runtime restart, first re-enter the repo directory:

```python
%cd /content/group5-rec-engines
```

Then run each notebook with:

```python
!jupyter nbconvert --to notebook --execute --inplace notebooks/01_non_personalized.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/02_collaborative_filtering.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/03_content_based.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/04_context_aware.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/05_evaluation.ipynb
```

Run them in order (01 → 05) the first time, since notebook 05 loads saved results from notebooks 01–04.

**Note:** Each notebook can also be run independently — they all load data fresh. But notebook 05 requires the `.pkl` result files that notebooks 01–04 save to `results/`.

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
