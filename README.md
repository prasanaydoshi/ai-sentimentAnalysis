# ai-sentimentAnalysis
# Reddit + Markets Sentiment Analytics

A production‑ready pipeline that collects finance/tech chatter from Reddit, cleans it, saves it to SQLite, and visualizes AI-enhanced sentiment and simple price correlations in a Streamlit dashboard.

## ✨ Features
- **Reddit ingestion (PRAW)** across curated subreddits for stocks/brands/crypto mentions.
- **Data cleaning & schema** with safe defaults; upserts to prevent duplicates.
- **Optional AI sentiment** (Hugging Face / PyTorch) with batching (Apple‑Silicon friendly).
- **Fast Streamlit dashboard** with cached reads, asset tracking, and correlation views.
- **Simple storage**: SQLite (`data/sentiment.db`) + Parquet backups for raw JSON.

## 📂 Repo Structure
```
.
├─ app/
│  └─ dashboard.py                 # Streamlit app (entrypoint)
├─ ingestion/
│  ├─ orchestrator.py              # CLI runner (once/scheduler, optional AI)
│  ├─ reddit_pull.py               # Reddit client + collectors
│  ├─ data_cleaner.py              # Text/record cleanup
│  └─ storage.py                   # SQLite + Parquet persistence
├─ data/
│  ├─ sentiment.db                 # (generated) SQLite DB
│  └─ raw/                         # (generated) raw post parquet files
├─ .streamlit/
│  └─ secrets.toml                 # (created in cloud) Reddit/API keys (optional)
├─ requirements.txt
├─ config.yaml                     # (user‑supplied) assets + Reddit creds
├─ run_dashboard.py                # local helper to launch Streamlit
└─ README.md
```

> **Heads‑up:** `run_dashboard.py` launches `app/dashboard.py`. If your app file is named differently, update either the filename or the runner (see “Known gotchas”).

## 🧰 Technical Specs (from code)
- **DB path default:** `data/sentiment.db` constructed/created on init.  
- **Dashboard DB discovery:** honors `SENTIMENT_DB` env var, else uses the default.  
- **Dashboard cache:** reads are cached with `@st.cache_data(ttl=300)` (5 minutes).  
- **Orchestrator CLI:** `--mode {once|scheduler}`, `--with-ai`, `--model`, `--batch-size`, `--config`.  
- **Creds validation:** startup check fails with placeholder Reddit creds.  
- **AI mode files:** expects `models/sentiment_analyzer.py` & `models/stock_data.py` on `PYTHONPATH`.

## 🚀 Quickstart (Local)

1) **Clone & create env**
```bash
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

2) **Create `config.yaml`** 

3) **Run a single collection cycle** (fills `data/sentiment.db`)
```bash
python ingestion/orchestrator.py --mode once                # light (no AI)
# Optional AI mode (heavier): add --with-ai --model cardiffnlp/twitter-roberta-base-sentiment-latest
```

4) **Launch the dashboard**
```bash
python run_dashboard.py
# or
streamlit run app/dashboard.py --server.runOnSave true
```


> **Cache/refresh behavior:** Each viewer runs the app anew, but data loads are cached for ~5 minutes. If the DB file changes underneath, the dashboard reflects it after cache expiry or a manual rerun (Menu → Rerun).


## 🧪 How the Dashboard Loads & Caches Data
- DB path: `SENTIMENT_DB` env var → else `data/sentiment.db`.
- Caching: 5‑minute TTL to reduce disk I/O.
- If the DB is missing, the app shows a friendly message with CLI steps to populate it.

## 🧩 Known gotchas & fixes
- **Entrypoint mismatch:** `run_dashboard.py` runs `app/dashboard.py`. If your file is named `app/ai_dashboard.py`, either rename it to `dashboard.py` or edit the runner (and Streamlit app path) accordingly.
- **Missing Reddit creds:** the orchestrator validates `config.yaml`; placeholders will stop the run.
- **AI model downloads:** Some Hugging Face models are heavy and can fail on free-tier bandwidth/memory. Start with a lighter model or run AI locally and only ship scored data to the cloud.

## 📈 Table Schema (SQLite)
Created automatically on first write. Key columns include:
- `id` (PK), `text`, `cleaned_text`, `created_at`, `processed_at`, `source`, `author`, `author_id`, `language`.
- `metrics`, `user_metrics` (JSON as TEXT), `tickers`, `hashtags` (JSON as TEXT).
- Optional AI: `sentiment_label`, `sentiment_score`, `sentiment_confidence`.
- Reddit-specific: `title`, `subreddit`, `url`.
- `raw_data` (original post JSON).

---

Made with ❤️ by Prasana.
