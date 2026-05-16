# Sentiment Classifier

End-to-end NLP pipeline: IMDB review classification → REST API → Docker deployment.

## Problem Statement
Given a movie review as raw text, predict whether the sentiment is **positive** or **negative**.

## Results
| Model | Accuracy | F1 Score |
|---|---|---|
| Random baseline | 50.0% | — |
| TF-IDF + Logistic Regression | ~89% | ~0.89 |
| DistilBERT (fine-tuned) | ~93% | ~0.93 |

## Project Structure
```
sentiment-classifier/
├── configs/config.yaml      # All hyperparameters in one place
├── data/
│   ├── raw/                 # Downloaded data (gitignored)
│   └── processed/           # Cleaned data (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory data analysis
│   ├── 02_baseline_model.ipynb
│   └── 03_bert_model.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py        # Downloads IMDB from HuggingFace
│   │   └── preprocessor.py  # Text cleaning pipeline
│   ├── models/
│   │   ├── train.py         # Training + MLflow logging
│   │   └── evaluate.py      # Metrics computation
│   └── api/
│       ├── main.py          # FastAPI app
│       └── schema.py        # Request/response schemas
├── tests/                   # Unit + integration tests
├── Dockerfile
├── docker-compose.yml
└── Makefile                 # Common commands
```

## Quick Start
```bash
# 1. Setup environment
make setup

# 2. Download data and train baseline
make data
make train

# 3. Run API locally
make run
# → http://localhost:8000/docs

# 4. Test the endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely brilliant!"}'

# 5. Run tests
make test
```

## API Usage
```bash
POST /predict
{"text": "Great film, highly recommend!"}

→ {"label": "positive", "confidence": 0.97, "label_id": 1}
```

## Dataset
- **Source:** IMDB Movie Reviews (HuggingFace Datasets)
- **Size:** 25,000 train / 25,000 test
- **Balance:** Perfectly balanced (50% positive, 50% negative)

## Tech Stack
`Python` `scikit-learn` `HuggingFace Transformers` `FastAPI` `Docker` `MLflow`