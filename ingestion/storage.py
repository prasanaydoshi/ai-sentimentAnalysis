# ingestion/storage.py
import sqlite3
import pandas as pd
import os
from datetime import datetime, timezone
import logging
import json
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class DataStorage:
    """
    Data storage layer with SQLite for MVP (easily upgradeable to Postgres).
    Design choice: Hybrid approach - SQLite for structured data, Parquet for raw JSON.
    """
    
    def __init__(self, db_path: str = "data/sentiment.db", parquet_dir: str = "data/raw"):
        self.db_path = db_path
        self.parquet_dir = parquet_dir
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(parquet_dir, exist_ok=True)
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            # Posts table - main data store
            conn.execute("""
                CREATE TABLE IF NOT EXISTS posts (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    cleaned_text TEXT,
                    created_at TIMESTAMP,
                    processed_at TIMESTAMP,
                    source TEXT NOT NULL,
                    author_id TEXT,
                    author TEXT,  -- Reddit author name
                    language TEXT,
                    
                    -- Metrics (JSON for flexibility)
                    metrics TEXT,
                    user_metrics TEXT,
                    
                    -- Extracted features
                    tickers TEXT,  -- JSON array of extracted tickers
                    hashtags TEXT, -- JSON array of extracted hashtags
                    
                    -- Sentiment scores (populated by model pipeline)
                    sentiment_label TEXT,
                    sentiment_score REAL,
                    sentiment_confidence REAL,
                    
                    -- Reddit-specific fields
                    title TEXT,  -- Reddit post title
                    subreddit TEXT,  -- for Reddit posts
                    url TEXT,
                    
                    -- Twitter/general fields
                    raw_data TEXT  -- JSON of original data
                )
            """)
            
            # Aggregated metrics table for dashboard performance
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hourly_sentiment (
                    timestamp TIMESTAMP,
                    asset TEXT,
                    source TEXT,
                    positive_count INTEGER DEFAULT 0,
                    negative_count INTEGER DEFAULT 0,
                    neutral_count INTEGER DEFAULT 0,
                    avg_sentiment REAL,
                    total_posts INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (timestamp, asset, source)
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON posts(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON posts(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sentiment ON posts(sentiment_label)")
            
            logger.info("Database initialized successfully")
    
    def save_posts(self, df: pd.DataFrame, save_raw: bool = True) -> int:
        """
        Save processed posts to database and optionally raw data to Parquet.
        Design choice: Upsert to handle duplicate posts gracefully.
        """
        if df.empty:
            return 0
        
        # Prepare data for database
        db_df = df.copy()
        
        # Convert lists to JSON strings for SQLite storage
        json_columns = ['extracted_tickers', 'extracted_hashtags', 'metrics', 'user_metrics', 'raw_data']
        for col in json_columns:
            if col in db_df.columns:
                db_df[col] = db_df[col].apply(lambda x: json.dumps(x) if x is not None else '{}')
        
        # Ensure all required columns exist with defaults
        required_columns = {
            'title': '',
            'author': '',
            'author_id': '',
            'subreddit': '',
            'url': '',
            'language': 'en',
            'sentiment_label': None,
            'sentiment_score': None,
            'sentiment_confidence': None,
            'raw_data': '{}',
            'metrics': '{}',
            'user_metrics': '{}',
            'extracted_tickers': '[]',
            'extracted_hashtags': '[]'
        }
        
        for col, default_val in required_columns.items():
            if col not in db_df.columns:
                db_df[col] = default_val
        
        # Get existing table columns to only include compatible ones
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("PRAGMA table_info(posts)")
                existing_columns = [row[1] for row in cursor.fetchall()]
                
                # Only keep columns that exist in the table
                db_df = db_df[[col for col in db_df.columns if col in existing_columns]]
                
                logger.info(f"Saving columns: {list(db_df.columns)}")
                
                # Use REPLACE to handle duplicates
                db_df.to_sql('posts', conn, if_exists='append', index=False, method='multi')
            
            # Save raw data to Parquet for backup/debugging
            if save_raw:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                parquet_path = os.path.join(self.parquet_dir, f"raw_posts_{timestamp}.parquet")
                df.to_parquet(parquet_path, compression='snappy')
                logger.info(f"Raw data saved to {parquet_path}")
            
            logger.info(f"Saved {len(df)} posts to database")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error saving posts to database: {e}")
            logger.error(f"DataFrame columns: {list(db_df.columns)}")
            logger.error(f"Database table columns: {existing_columns if 'existing_columns' in locals() else 'unknown'}")
            return 0
    
    def get_recent_posts(self, hours: int = 24, asset: Optional[str] = None) -> pd.DataFrame:
        """Retrieve recent posts for dashboard display."""
        query = """
            SELECT * FROM posts 
            WHERE created_at > datetime('now', '-{} hours')
        """.format(hours)
        
        if asset:
            # Search in tickers JSON array
            query += f" AND (tickers LIKE '%{asset}%' OR hashtags LIKE '%{asset.lower()}%')"
        
        query += " ORDER BY created_at DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn)
        
        return df
    
    def get_sentiment_aggregates(self, hours: int = 24) -> pd.DataFrame:
        """Get aggregated sentiment data for dashboard charts."""
        query = """
            SELECT 
                strftime('%Y-%m-%d %H:00:00', created_at) as hour,
                source,
                sentiment_label,
                COUNT(*) as count,
                AVG(sentiment_score) as avg_score
            FROM posts 
            WHERE created_at > datetime('now', '-{} hours')
            AND sentiment_label IS NOT NULL
            GROUP BY hour, source, sentiment_label
            ORDER BY hour DESC
        """.format(hours)
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn)
        
        return df