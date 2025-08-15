# ingestion/orchestrator.py
"""
Standalone Reddit data collection orchestrator with Apple Silicon fixes.
"""

from __future__ import annotations

import os
import sys
import time
import logging
import argparse
import gc
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

import yaml
import schedule

# ---------------------------
# Path setup (Option B)
# ---------------------------
SCRIPT_DIR: Path = Path(__file__).resolve().parent             # .../sentiment analysis/ingestion
PROJECT_ROOT: Path = SCRIPT_DIR.parent                         # .../sentiment analysis
INGESTION_DIR: Path = SCRIPT_DIR
MODELS_DIR: Path = PROJECT_ROOT / "models"

# Ensure we can import sibling modules
for p in (INGESTION_DIR, MODELS_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

# Ingestion-layer imports (always available)
from reddit_pull import RedditIngestion
from data_cleaner import DataCleaner
from storage import DataStorage

# ---------------------------
# Apple Silicon Memory Optimization
# ---------------------------
def setup_apple_silicon_optimizations():
    """Configure optimal settings for Apple Silicon."""
    try:
        import torch
        # Limit threads to prevent memory issues
        torch.set_num_threads(1)
        
        # Set conservative memory management
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('medium')
            
        # Aggressive garbage collection
        gc.set_threshold(100, 10, 10)
        
        logger.info("✅ Apple Silicon optimizations applied")
    except ImportError:
        logger.info("PyTorch not available, skipping optimizations")
    except Exception as e:
        logger.warning(f"Could not apply optimizations: {e}")

# ---------------------------
# Logging
# ---------------------------
LOG_FILE = PROJECT_ROOT / "reddit_collection.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ],
)
logger = logging.getLogger("reddit_orchestrator")


# ---------------------------
# Config helpers (unchanged)
# ---------------------------
def find_config_path(explicit: Optional[str] = None) -> Path:
    """
    Priority:
      1) --config <path>
      2) CONFIG_PATH env var
      3) PROJECT_ROOT/config.yaml
      4) SCRIPT_DIR/config.yaml
    Returns first existing; else PROJECT_ROOT/config.yaml (for error messaging).
    """
    candidates: list[Path] = []

    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())

    env_p = os.getenv("CONFIG_PATH")
    if env_p:
        candidates.append(Path(env_p).expanduser().resolve())

    candidates.extend([
        PROJECT_ROOT / "config.yaml",
        SCRIPT_DIR / "config.yaml",
    ])

    for c in candidates:
        if c.exists():
            return c

    return PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path) -> Dict:
    try:
        logger.info(f"Loading config from: {config_path}")
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.error(f"Config not found at: {config_path}")
        logger.error(f"Working directory: {os.getcwd()}")
        logger.error(f"PROJECT_ROOT contents: {sorted(p.name for p in PROJECT_ROOT.iterdir())}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        sys.exit(1)


def check_setup(cfg: Dict, cfg_path: Path) -> bool:
    if not cfg:
        logger.error("Config is empty. Populate your config.yaml.")
        return False

    reddit_cfg = cfg.get("reddit", {}) or {}
    cid = reddit_cfg.get("client_id", "")
    csec = reddit_cfg.get("client_secret", "")

    if (not cid or not csec or
        "YOUR_CLIENT_ID_HERE" in cid or "YOUR_CLIENT_SECRET_HERE" in csec):
        logger.error("Reddit API credentials missing/placeholders.")
        logger.error(f"Edit: {cfg_path}")
        logger.error("Create creds at https://www.reddit.com/prefs/apps/")
        return False

    logger.info("Configuration looks good.")
    return True


# ---------------------------
# Memory-Safe Sentiment Analysis Helper
# ---------------------------
# Improved safe_sentiment_analysis function
# Replace the existing function in your orchestrator.py

# Complete fix - replace both functions in your orchestrator.py

def safe_sentiment_analysis(sentiment_analyzer, df, text_column="cleaned_text", batch_size=5):
    """
    Process sentiment analysis in small batches with robust error handling.
    Fixes: tensor size issues, index duplication, text truncation.
    """
    import pandas as pd
    
    logger.info(f"Starting batched sentiment analysis: {len(df)} posts in batches of {batch_size}")
    
    # Reset the input dataframe index to avoid issues
    df_clean = df.reset_index(drop=True).copy()
    
    results = []
    total_batches = (len(df_clean) + batch_size - 1) // batch_size
    
    for i in range(0, len(df_clean), batch_size):
        batch_num = i // batch_size + 1
        batch_df = df_clean.iloc[i:i+batch_size].copy()
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_df)} posts)")
        
        try:
            # Extract and properly truncate texts for analysis
            texts = []
            for text in batch_df[text_column].tolist():
                if isinstance(text, str):
                    # More aggressive truncation - RoBERTa max tokens ~512
                    if len(text) > 1500:  # Conservative character limit
                        text = text[:1500] + "..."
                    texts.append(text)
                else:
                    texts.append("")
            
            # Analyze this batch
            batch_results = sentiment_analyzer.analyze_texts(texts)
            
            # Convert results to DataFrame with clean index
            if isinstance(batch_results, list):
                batch_sentiment_df = pd.DataFrame(batch_results)
            else:
                batch_sentiment_df = batch_results.copy()
            
            # Ensure clean indices
            batch_sentiment_df = batch_sentiment_df.reset_index(drop=True)
            batch_df = batch_df.reset_index(drop=True)
            
            # Handle length mismatches
            if len(batch_sentiment_df) != len(batch_df):
                logger.warning(f"Batch {batch_num}: result mismatch ({len(batch_sentiment_df)} vs {len(batch_df)})")
                
                if len(batch_sentiment_df) < len(batch_df):
                    # Pad with neutral sentiment
                    padding_needed = len(batch_df) - len(batch_sentiment_df)
                    neutral_rows = [{
                        'sentiment_label': 'neutral',
                        'sentiment_score': 0.0,
                        'confidence': 0.0,
                        'positive_prob': 0.0,
                        'negative_prob': 0.0,
                        'neutral_prob': 1.0,
                    } for _ in range(padding_needed)]
                    padding_df = pd.DataFrame(neutral_rows)
                    batch_sentiment_df = pd.concat([batch_sentiment_df, padding_df], ignore_index=True)
                else:
                    # Truncate to match
                    batch_sentiment_df = batch_sentiment_df.head(len(batch_df)).reset_index(drop=True)
            
            # Combine original data with sentiment results using concat with axis=1
            batch_combined = pd.concat([batch_df, batch_sentiment_df], axis=1, ignore_index=False)
            
            # Store result with a fresh index
            batch_combined = batch_combined.reset_index(drop=True)
            results.append(batch_combined)
            
            # Memory cleanup
            del batch_df, batch_sentiment_df, batch_combined, texts, batch_results
            gc.collect()
            
            logger.info(f"✅ Batch {batch_num} completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error in batch {batch_num}: {e}")
            # Add fallback data with error sentiment
            batch_df = batch_df.reset_index(drop=True)
            batch_df['sentiment_label'] = 'error'
            batch_df['sentiment_score'] = 0.0
            batch_df['confidence'] = 0.0
            batch_df['positive_prob'] = 0.0
            batch_df['negative_prob'] = 0.0
            batch_df['neutral_prob'] = 1.0
            results.append(batch_df)
            continue
    
    if results:
        # Final concatenation with completely fresh index
        try:
            final_df = pd.concat(results, ignore_index=True)
            logger.info(f"✅ Sentiment analysis complete: {len(final_df)} posts processed")
            return final_df
        except Exception as e:
            logger.error(f"Error concatenating results: {e}")
            # Fallback: return original with default sentiment
            return add_default_sentiment_columns(df_clean)
    else:
        logger.error("❌ All sentiment analysis batches failed")
        return add_default_sentiment_columns(df_clean)


def add_default_sentiment_columns(df):
    """Add default sentiment columns to a dataframe."""
    df_copy = df.copy()
    df_copy['sentiment_label'] = 'neutral'
    df_copy['sentiment_score'] = 0.0
    df_copy['confidence'] = 0.5
    df_copy['positive_prob'] = 0.0
    df_copy['negative_prob'] = 0.0
    df_copy['neutral_prob'] = 1.0
    return df_copy


# Also add this method to handle duplicate posts in your AIOrchestrator class
# Add this to the AIOrchestrator.collect_reddit_data method, right before saving:

# ---------------------------
# Orchestrators
# ---------------------------
class BaseOrchestrator:
    """Coordinates Reddit ingestion, cleaning, and storage."""

    def __init__(self, cfg: Dict):
        self.config: Dict = cfg
        logger.info("Initializing base components...")
        self.reddit = RedditIngestion(self.config)
        self.cleaner = DataCleaner()
        self.storage = DataStorage()

        assets = self.config.get("assets", {}) or {}
        self.all_assets = list(assets.get("tickers", [])) + \
                          list(assets.get("brands", [])) + \
                          list(assets.get("crypto", []))

        poll = self.config.get("settings", {}).get("reddit_poll_minutes", 30)
        try:
            self.poll_minutes = int(poll)
        except Exception:
            self.poll_minutes = 30

        logger.info(f"Tracking {len(self.all_assets)} assets: {self.all_assets}")
        logger.info(f"Poll interval: {self.poll_minutes} minute(s)")

    def collect_reddit_data(self) -> None:
        start = datetime.now()
        logger.info("=" * 60)
        logger.info("Starting Reddit collection cycle")
        logger.info("=" * 60)

        try:
            if not self.all_assets:
                logger.warning("No assets configured in config.yaml.")
                return

            logger.info(f"Searching Reddit for: {', '.join(self.all_assets)}")
            raw_df = self.reddit.run_collection_cycle(self.all_assets)

            if raw_df is None or raw_df.empty:
                logger.info("No Reddit data collected this cycle.")
                return

            logger.info(f"Collected {len(raw_df)} raw posts. Cleaning...")
            processed_df = self.cleaner.process_dataframe(raw_df)

            if processed_df is None or processed_df.empty:
                logger.warning("No posts remaining after cleaning.")
                return

            logger.info(f"Saving {len(processed_df)} posts to database...")
            saved_count = self.storage.save_posts(processed_df)

            duration = (datetime.now() - start).total_seconds()
            logger.info("-" * 40)
            logger.info("COLLECTION SUMMARY")
            logger.info(f"  Raw posts collected : {len(raw_df)}")
            logger.info(f"  Posts after cleaning: {len(processed_df)}")
            logger.info(f"  Posts saved to DB   : {saved_count}")
            logger.info(f"  Duration (sec)      : {duration:.1f}")
            logger.info("-" * 40)

        except Exception as e:
            logger.error(f"Reddit collection failed: {e}", exc_info=True)

    def _db_path(self) -> Path:
        dp = getattr(self.storage, "db_path", None)
        try:
            return Path(dp).resolve() if dp else PROJECT_ROOT / "data" / "sentiment.db"
        except Exception:
            return PROJECT_ROOT / "data" / "sentiment.db"

    def run_once(self) -> None:
        self.collect_reddit_data()
        # Quick DB stats
        try:
            import sqlite3
            db_path = self._db_path()
            if not db_path.exists():
                logger.info(f"DB not found at {db_path} (stats skipped).")
                return
            with sqlite3.connect(str(db_path)) as conn:
                total_posts = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
                unique_subs = conn.execute(
                    "SELECT COUNT(DISTINCT subreddit) FROM posts WHERE subreddit IS NOT NULL"
                ).fetchone()[0]
            logger.info(f"Database now contains {total_posts} posts across {unique_subs} subreddits. [{db_path}]")
        except Exception as e:
            logger.error(f"Error checking database stats: {e}")

    def setup_scheduler(self) -> None:
        schedule.every(self.poll_minutes).minutes.do(self.collect_reddit_data)
        logger.info(f"Scheduler configured to run every {self.poll_minutes} minute(s).")

    def run_scheduler(self) -> None:
        self.setup_scheduler()
        logger.info("Running initial collection before scheduler starts...")
        self.run_once()
        logger.info("Starting scheduler. Press Ctrl+C to stop.")
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user.")


class AIOrchestrator(BaseOrchestrator):
    """Extends BaseOrchestrator with memory-safe AI sentiment & correlation."""

    def __init__(self, cfg: Dict, model_name: Optional[str] = None):
        # Apply Apple Silicon optimizations early
        setup_apple_silicon_optimizations()
        
        super().__init__(cfg)
        # Lazy imports so that --with-ai is optional
        try:
            from sentiment_analyzer import SentimentAnalyzer, FinancialSentimentEnhancer
            from stock_data import StockDataProvider, SentimentPriceCorrelator
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "AI modules not found. Ensure models/ is on sys.path and files exist:\n"
                " - models/sentiment_analyzer.py\n - models/stock_data.py\n"
                f"Underlying error: {e}"
            )

        self.sentiment_analyzer = SentimentAnalyzer(model_name=model_name)
        self.sentiment_enhancer = FinancialSentimentEnhancer()
        self.stock_provider = StockDataProvider()
        self.correlator = SentimentPriceCorrelator(self.stock_provider)

    def filter_new_posts(self, df):
        """Filter out posts that already exist in the database."""
        try:
            import sqlite3
            db_path = self._db_path()
            
            if not db_path.exists():
                return df  # No database yet, all posts are new
            
            with sqlite3.connect(str(db_path)) as conn:
                # Get existing post IDs
                existing_ids = set()
                try:
                    cursor = conn.execute("SELECT id FROM posts")
                    existing_ids = {row[0] for row in cursor.fetchall()}
                except Exception:
                    # Table might not exist yet
                    return df
            
            if not existing_ids:
                return df  # No existing posts
            
            # Filter out existing posts
            new_posts = df[~df['id'].isin(existing_ids)]
            
            if len(new_posts) < len(df):
                logger.info(f"Filtered out {len(df) - len(new_posts)} duplicate posts. {len(new_posts)} new posts remain.")
            
            return new_posts
            
        except Exception as e:
            logger.warning(f"Error filtering duplicate posts: {e}. Proceeding with all posts.")
            return df

    def collect_reddit_data(self) -> None:
        """Override: after cleaning, run AI scoring & correlation with memory safety."""
        start = datetime.now()
        logger.info("=" * 60)
        logger.info("Starting AI-enhanced Reddit collection cycle")
        logger.info("=" * 60)

        try:
            if not self.all_assets:
                logger.warning("No assets configured in config.yaml.")
                return

            logger.info(f"Searching Reddit for: {', '.join(self.all_assets)}")
            raw_df = self.reddit.run_collection_cycle(self.all_assets)
            if raw_df is None or raw_df.empty:
                logger.info("No Reddit data collected this cycle.")
                return

            logger.info(f"Collected {len(raw_df)} raw posts. Cleaning...")
            processed_df = self.cleaner.process_dataframe(raw_df)
            if processed_df is None or processed_df.empty:
                logger.warning("No posts remaining after cleaning.")
                return

            # ---- MEMORY-SAFE AI sentiment ----
            logger.info("Running memory-safe AI sentiment analysis...")
            try:
                # Use smaller batch size for Apple Silicon
                batch_size = 3 if len(processed_df) > 50 else 5
                sentiment_df = safe_sentiment_analysis(
                    self.sentiment_analyzer, 
                    processed_df, 
                    text_column="cleaned_text",
                    batch_size=batch_size
                )

                logger.info("✅ Skipping financial enhancement (debug mode)")
                final_df = sentiment_df.copy()
                if 'confidence' in final_df.columns:
                    final_df['sentiment_confidence'] = final_df['confidence']
                    final_df = final_df.drop('confidence', axis=1)
                logger.info("✅ AI sentiment analysis completed successfully")
            except Exception as e:
                logger.error(f"AI sentiment failed, saving cleaned only: {e}", exc_info=True)
                final_df = add_default_sentiment_columns(processed_df)


            # Filter out duplicate posts before saving
            final_df = self.filter_new_posts(final_df)

            if final_df.empty:
                logger.info("No new posts to save after filtering duplicates.")
                return

            # Save
            logger.info(f"Saving {len(final_df)} posts to database...")
            saved_count = self.storage.save_posts(final_df)

            # Optional: correlations (best-effort, skip if memory issues)
            try:
                main_tickers = ["AAPL", "MSFT", "NVDA"]  # Reduced for memory
                for t in main_tickers:
                    _ = self.correlator.analyze_sentiment_price_correlation(
                        final_df, t, timeframe_hours=24
                    )
            except Exception as e:
                logger.warning(f"Correlation analysis skipped: {e}")

            duration = (datetime.now() - start).total_seconds()
            logger.info("-" * 40)
            logger.info("AI COLLECTION SUMMARY")
            logger.info(f"  Raw posts collected : {len(raw_df)}")
            logger.info(f"  Posts after cleaning: {len(processed_df)}")
            logger.info(f"  Posts saved to DB   : {saved_count}")
            logger.info(f"  Duration (sec)      : {duration:.1f}")
            logger.info("-" * 40)

        except Exception as e:
            logger.error(f"AI collection failed: {e}", exc_info=True)

# ---------------------------
# CLI
# ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reddit Sentiment Data Collector")
    p.add_argument("--config", type=str, default=None,
                   help="Path to config.yaml (overrides discovery & CONFIG_PATH).")
    p.add_argument("--mode", choices=["once", "scheduler"], default="once",
                   help="Run once or continuously with a scheduler.")
    p.add_argument("--with-ai", action="store_true",
                   help="Enable AI sentiment & correlation (requires models/* and deps).")
    p.add_argument("--model", type=str, default=None,
                   help="HF model id (e.g. 'distilbert-base-uncased-finetuned-sst-2-english').")
    p.add_argument("--batch-size", type=int, default=5,
                   help="Batch size for sentiment analysis (smaller = less memory usage).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = find_config_path(args.config)
    cfg = load_config(cfg_path)

    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"SCRIPT_DIR: {SCRIPT_DIR}")
    logger.info(f"PROJECT_ROOT: {PROJECT_ROOT}")
    logger.info(f"Using config: {cfg_path}")

    if not check_setup(cfg, cfg_path):
        print("\n❌ Setup incomplete. Please fix the issues above.")
        sys.exit(1)

    # Choose orchestrator
    if args.with_ai:
        try:
            orch = AIOrchestrator(cfg, model_name=args.model)
            logger.info(f"AI mode enabled. Model: {args.model or 'default in sentiment_analyzer.py'}")
            logger.info(f"Batch size: {args.batch_size} (use --batch-size to adjust)")
        except RuntimeError as e:
            logger.error(str(e))
            sys.exit(1)
    else:
        orch = BaseOrchestrator(cfg)

    if args.mode == "once":
        orch.run_once()
    else:
        orch.run_scheduler()


if __name__ == "__main__":
    main()