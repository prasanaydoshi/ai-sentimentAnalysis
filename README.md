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

2) **Create `config.yaml`** (see template below).

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

## ☁️ Deploy to Streamlit Cloud

1) **Push to GitHub** with this structure and include `requirements.txt`.
2) In Streamlit Cloud: **New app → Connect repo → Branch → File = `app/dashboard.py`**.
3) (Optional) **Secrets** → add Reddit keys (only needed if you plan to run ingestion elsewhere with repo secrets; dashboard itself only reads the SQLite file).
4) **Boot time**: heavy ML wheels (torch/transformers) can slow cold starts. If you only visualize pre‑computed sentiment, you can remove the AI libs from `requirements.txt` to speed up deploys.
5) **Data updates**: Streamlit Cloud doesn’t run background jobs. Use one of:
   - **GitHub Action (recommended):** run `ingestion/orchestrator.py` on a schedule, commit updated `data/sentiment.db` back to the repo, and the app will read the new file.
   - **External DB:** adapt `storage.py` to Postgres (Neon/Supabase); point the app to it.
   - **Manual:** run locally and push the refreshed DB.

> **Cache/refresh behavior:** Each viewer runs the app anew, but data loads are cached for ~5 minutes. If the DB file changes underneath, the dashboard reflects it after cache expiry or a manual rerun (Menu → Rerun).

## 🔐 Config & Secrets

### `config.yaml` (template)
```yaml
reddit:
  client_id: YOUR_CLIENT_ID
  client_secret: YOUR_CLIENT_SECRET
  user_agent: reddit-sentiment-app/0.1 by yourname

assets:
  tickers: [AAPL, MSFT, NVDA, TSLA, GOOGL, AMZN, META]
  brands:  [Apple, Microsoft, Nvidia, Tesla, Google, Amazon, Meta]
  crypto:  [BTC, ETH, bitcoin, ethereum, crypto, cryptocurrency]

settings:
  reddit_poll_minutes: 30
```

### `.streamlit/secrets.toml` (OPTIONAL; for Cloud)
```toml
[reddit]
client_id = "YOUR_CLIENT_ID"
client_secret = "YOUR_CLIENT_SECRET"
user_agent = "reddit-sentiment-app/0.1 by yourname"
```

> If you prefer secrets over a tracked `config.yaml`, adjust your orchestrator to read from `st.secrets` or environment variables. The current code reads `config.yaml` by default.

## 🤖 GitHub Action (scheduled ingestion)

Add `.github/workflows/ingest.yml`:
```yaml
name: Ingest Reddit Sentiment
on:
  schedule:
    - cron: "*/30 * * * *"  # every 30 minutes
  workflow_dispatch: {}

permissions:
  contents: write

jobs:
  ingest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -U pip && pip install -r requirements.txt
      - name: Write config from secrets
        run: |
          cat > config.yaml <<'YAML'
          reddit:
            client_id: ${{ secrets.REDDIT_CLIENT_ID }}
            client_secret: ${{ secrets.REDDIT_CLIENT_SECRET }}
            user_agent: reddit-sentiment-app/0.1 by ${{ github.repository_owner }}
          assets:
            tickers: [AAPL, MSFT, NVDA]
            brands:  [Apple, Microsoft, Nvidia]
            crypto:  [BTC, ETH, bitcoin]
          settings:
            reddit_poll_minutes: 30
          YAML
      - name: Run collector (no AI; faster/cheaper)
        run: python ingestion/orchestrator.py --mode once
      - name: Commit DB updates
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add data/sentiment.db data/raw || true
          git commit -m "chore(data): refresh sentiment DB [skip ci]" || echo "No changes"
          git push
```

> If you commit large DB files frequently, consider Git LFS or switching to Postgres.

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

## 🗺️ Roadmap
- Switch from SQLite → Postgres for shared, multi‑writer storage.
- Add ingestion for Twitter/X, Hacker News, StockTwits.
- Add Streamlit “Refresh now” button to clear cache on demand.
- Ship a small sample DB for instant demo.

---

Made with ❤️ by Prasana.
