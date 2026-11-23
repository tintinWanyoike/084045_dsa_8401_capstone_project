# Time-to-Dormancy Early Warning for Bank Accounts

DSA 8401 Applied Machine Learning capstone. For each active client on a monthly scoring
date, the model will predict the probability of becoming dormant (no transactions for N
consecutive days) within the next 30, 60 and 90 days.

One pipeline notebook builds and evaluates the model; a FastAPI service in a Docker container
serves it, with reason codes for every score and a drift check for each scoring run.

## Repository layout

```
notebook.ipynb
  the whole pipeline, Parts 1-6:
                                1 data collection   2 cleaning   3 EDA and dormancy window
                                4 labels, features, time split, audits   5 models and test evaluation
                                6 SHAP, drift monitoring, calibration, N = 60 check
data/
  raw/                        downloaded archives (not committed)
  interim/                    extracted tables and cleaned Parquet files (not committed)
  processed/                  modelling panel and feature dictionary (not committed)
models/                       served model (lgbm_hazard.txt), comparison models, metadata.json,
                              monitoring_reference.json (drift bins, alert rules, risk tiers),
service/
  app/                        FastAPI service: features.py, model.py, schemas.py, main.py
  tests/                      parity tests (API = notebooks) and endpoint tests, with fixtures
  examples/score_request.json example request (a real anonymised client)
Dockerfile, docker-compose.yml  container for the API
.github/workflows/ci.yml      runs the tests and builds the image on every push
reports/
  test_results.csv            test metrics for every model, horizon and subset
  monitoring/                 Evidently drift report, PSI by date, calibration, N = 60 sensitivity
run_pipeline.sh               runs every notebook top to bottom
requirements.txt
feature_dictionary.csv
```

## Setup

Python 3.11 or later. About 8 GB of free disk space (2 GB download, 2 GB extracted,
2 GB of cleaned files). 8 GB of RAM is enough: heavy steps stream data in batches.

```bash
git clone <repo-url> dormancy-capstone
cd dormancy-capstone
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

Interactively, top to bottom:

```bash
jupyter lab notebook.ipynb
```

Or reproduce everything from scratch, headless:

```bash
bash run_pipeline.sh             
```

Each notebook can also be run on its own with:

| Part | Runtime (2 CPUs) | Produces |
|---|---|---|
| 1 Data collection | ~1 min plus download | `data/raw/*.tar.gz`, `data/interim/extracted/` |
| 2 Data preparation | ~2 min | `data/interim/{trx,geo,dialog}_clean.parquet`, `clients.parquet` |
| 3 EDA | ~1 min | charts and decisions only |
| 4 Labels and features | ~1 min | `data/processed/panel.parquet`, `feature_dictionary.csv`, `data/interim/daily_activity.parquet` |
| 5 Modelling | ~7 min | `models/*`, `reports/test_results.csv`, `reports/tuning_results.csv` |
| 6 Explainability and monitoring | ~5 min | `models/monitoring_reference.json`, `models/feature_dictionary.csv`, `reports/monitoring/*` |

Each part reloads what it needs from the files written by earlier parts, so after a full
run any single part can be re-run on its own.ollection | ~1 min plus download | `data/raw/*.tar.gz`, `data/interim/extracted/` |
| 02 Data preparation | ~2 min | `data/interim/{trx,geo,dialog}_clean.parquet`, `clients.parquet` |
| 03 EDA | ~1 min | charts and decisions only (no files) |
| 04 Labels and features | ~1 min | `data/processed/panel.parquet`, `feature_dictionary.csv`, `data/interim/daily_activity.parquet` |
| 05 Modelling | ~7 min (2 CPUs) | `models/*`, `reports/test_results.csv`, `reports/tuning_results.csv` |
| 06 Explainability and monitoring | ~5 min | `models/monitoring_reference.json`, `models/feature_dictionary.csv`, `reports/monitoring/*` |

## The API

### Run it

With Docker (nothing else to install):

```bash
docker compose up --build          # or: docker build -t dormancy-api . && docker run --rm -p 8000:8000 dormancy-api
```

Without Docker, from the repository root:

```bash
pip install -r service/requirements.txt
uvicorn service.app.main:app --reload
```

Interactive documentation, where every endpoint can be tried in the browser:
http://localhost:8000/docs

### Endpoints

| Method and path | What it does |
|---|---|
| `GET /health` | Service status and model version |
| `GET /model-info` | Dormancy window, horizons, training and test dates, test metrics, risk tiers, feature descriptions |
| `POST /score` | One client: raw transactions in, 30/60/90-day risk, risk tier and reason codes out |
| `POST /score/features` | Many clients at once from precomputed features (up to 50,000 rows) |
| `POST /monitor/drift` | One scoring run's features in, PSI per feature and for the score, plus alerts |

### Try it

```bash
curl localhost:8000/health

curl -X POST localhost:8000/score \
     -H "Content-Type: application/json" \
     -d @service/examples/score_request.json
```

The response gives the risk, a tier (the top 5% of training-period scores are "high",
90-day risk of 40.5% or more; the next 15% are "medium", 3.2% or more), the 30-day
interval that dominates the risk, and the features pushing it up or down. For the example
client (one transaction in the last 90 days, 16 quiet days), shortened:

```json
{
  "status": "scored",
  "message": "Medium risk: 34.3% chance of 90 quiet days within the next 90 days.",
  "quiet_days": 16,
  "risk": {"p30": 0.0002, "p60": 0.0023, "p90": 0.3430},
  "risk_tier": "medium",
  "dominant_interval": 3,
  "reasons": [
    {"feature": "trx_90", "value": 1.0, "effect": "raises risk", "shap": 0.9853,
     "description": "Transactions in the last 90 days"},
    {"feature": "active_weeks_90", "value": 1.0, "effect": "raises risk", "shap": 0.9217,
     "description": "Weeks with any activity in the last 90 days (of 13)"}
  ],
  "model_version": "133013fbfdf1"
}
```

Input rules: times are bank-local (UTC+3; timezone-aware times are converted), only
transactions before `score_date` are used (later ones are counted and ignored), and the
full history is needed for the pause-history features. A client already 90+ days quiet
gets `already_dormant` and no score, because the model was never trained on them.

### Tests

```bash
pip install -r service/requirements-dev.txt
pytest service/tests -q
```

The parity tests use 40 real clients on two scoring dates (`service/tests/fixtures`) and
check that the API reproduces the notebook features and risk scores exactly. To rebuild the
fixtures after retraining: `python -m service.tests.make_fixtures` (needs the full data).

## Data

[MBD-mini](https://huggingface.co/datasets/ai-lab/MBD-mini) (Sber AI Lab, CC BY 4.0), a
10% client sample of the Multimodal Banking Dataset
([Mollaev et al., 2024](https://arxiv.org/abs/2409.17587)): 100,224 corporate clients,
38.8M transactions over January 2021 to December 2022, plus geo-activity and
support-dialogue embeddings. The clients are corporate customers of a non-Kenyan bank,
so results validate the method rather than describe Kenyan retail behaviour.
