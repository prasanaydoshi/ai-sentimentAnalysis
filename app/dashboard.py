# app/ai_dashboard.py
"""
AI-Enhanced Sentiment Dashboard with transformer/VADER analysis and stock correlations.
- Robust DB path handling (supports spaces in path; override with SENTIMENT_DB env var)
- Adaptive SELECT that matches whatever columns are actually present in 'posts'
- Enhanced with detailed asset tracking and collection insights
"""

from __future__ import annotations

import os
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import Counter
from typing import List, Dict

import streamlit as st
import pandas as pd
import plotly.express as px
import yaml

# ──────────────────────────────────────────────────────────────────────────────
# Paths & optional model imports
# ──────────────────────────────────────────────────────────────────────────────
# Ensure we can import from ../models
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

try:
    from stock_data import StockDataProvider
    from sentiment_analyzer import SentimentAnalyzer  # noqa: F401  (not used directly here)
    MODELS_AVAILABLE = True
except Exception:
    MODELS_AVAILABLE = False

# Absolute DB path (env override supported)
BASE_DIR = PROJECT_ROOT                    # .../sentiment analysis
DEFAULT_DB = BASE_DIR / "data" / "sentiment.db"
DB_PATH = Path(os.getenv("SENTIMENT_DB", str(DEFAULT_DB))).resolve()
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  
# ──────────────────────────────────────────────────────────────────────────────
# Streamlit page config & styles
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Sentiment Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .sentiment-positive { color: #00C851; font-weight: bold; }
    .sentiment-negative { color: #FF4444; font-weight: bold; }
    .sentiment-neutral  { color: #6C757D; font-weight: bold; }
    .alert-banner { 
        background: linear-gradient(90deg, #FF6B6B, #FF8E53);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-weight: bold;
    }
    .metric-positive { color: #00C851; }
    .metric-negative { color: #FF4444; }
    .asset-tracker {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

if not MODELS_AVAILABLE:
    st.warning("⚠️ AI add-ons not fully available. For stock correlation/smaller HF models: `pip install yfinance transformers torch`")

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def _normalize_json_like(val, want_list: bool = False):
    """Convert TEXT columns that may contain JSON to proper Python types."""
    if val is None:
        return [] if want_list else {}
    if isinstance(val, (list, dict)):
        return val
    s = str(val).strip()
    if s == "" or s in ("{}", "[]", "null", "None"):
        return [] if want_list else {}
    # If it looks like JSON, try to parse
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            parsed = json.loads(s)
            # Ensure correct type
            if want_list and isinstance(parsed, list):
                return parsed
            if not want_list and isinstance(parsed, dict):
                return parsed
            # Type mismatch: coerce
            return list(parsed) if want_list else dict(parsed)
        except Exception:
            pass
    # Fallbacks
    if want_list:
        # split on commas for simple cases like "AAPL, MSFT"
        return [x.strip() for x in s.split(",") if x.strip()]
    return {}

def _to_utc(dt_series: pd.Series) -> pd.Series:
    """Parse timestamps to UTC; coerce errors to NaT."""
    ser = pd.to_datetime(dt_series, utc=True, errors="coerce")
    # If parsed as naive (very rare due to utc=True), localize to UTC
    try:
        return ser.dt.tz_convert("UTC")
    except Exception:
        try:
            return ser.dt.tz_localize("UTC")
        except Exception:
            return ser

# ──────────────────────────────────────────────────────────────────────────────
# NEW: Asset tracking functions
# ──────────────────────────────────────────────────────────────────────────────
def load_config_assets() -> Dict:
    """Load the tracked assets from config.yaml"""
    try:
        config_path = PROJECT_ROOT / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('assets', {})
    except Exception as e:
        st.error(f"Could not load config: {e}")
    
    # Fallback default assets
    return {
        'tickers': ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL', 'AMZN', 'META'],
        'brands': ['Apple', 'Microsoft', 'Nvidia', 'Tesla', 'Google', 'Amazon', 'Meta'],
        'crypto': ['BTC', 'ETH', 'bitcoin', 'ethereum', 'crypto', 'cryptocurrency']
    }

def extract_asset_mentions(text: str, config_assets: Dict) -> List[str]:
    """Extract asset mentions from text"""
    if not text:
        return []
    
    text_lower = text.lower()
    mentions = []
    
    # Check all asset categories
    for category, assets in config_assets.items():
        for asset in assets:
            if asset.lower() in text_lower:
                mentions.append(asset.upper() if category == 'tickers' else asset)
    
    return list(set(mentions))  # Remove duplicates

# ──────────────────────────────────────────────────────────────────────────────
# Data loading (ORIGINAL)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_sentiment_data(db_file: Path = DB_PATH) -> pd.DataFrame:
    """
    Load posts with AI sentiment columns if they exist.
    Adapts to whichever columns are available in the 'posts' table.
    """
    if not Path(db_file).exists():
        return pd.DataFrame()

    try:
        with sqlite3.connect(str(db_file)) as conn:
            # Discover actual columns
            cols_df = pd.read_sql_query("PRAGMA table_info(posts);", conn)
            table_cols = set(cols_df["name"].tolist())

            # Core columns we try to bring if present
            core_cols: List[str] = [
                "id", "text", "cleaned_text", "created_at", "processed_at",
                "source", "author_id", "language", "tickers", "hashtags",
                "url", "subreddit"
            ]
            optional_cols: List[str] = [
                "author", "title", "metrics", "user_metrics",
                "raw_data", "sentiment_label", "sentiment_score",
                "sentiment_confidence", "confidence"  # allow either name
            ]

            selected: List[str] = [c for c in core_cols if c in table_cols]

            # Bring metrics/user_metrics if present
            for c in ("metrics", "user_metrics", "author", "title", "raw_data"):
                if c in table_cols:
                    selected.append(c)

            # Sentiment columns: allow confidence aliasing
            if "sentiment_label" in table_cols:
                selected.append("sentiment_label")
            if "sentiment_score" in table_cols:
                selected.append("sentiment_score")
            if "sentiment_confidence" in table_cols:
                selected.append("sentiment_confidence")
            elif "confidence" in table_cols:
                # alias only if original column exists
                selected.append("confidence AS sentiment_confidence")

            if not selected:
                # Table exists but no expected columns; fetch at least id/text/timestamps if present
                selected = [c for c in ("id", "text", "created_at", "processed_at") if c in table_cols]

            query = f"""
                SELECT {", ".join(selected)}
                FROM posts
                {"WHERE sentiment_label IS NOT NULL" if "sentiment_label" in table_cols else ""}
                ORDER BY created_at DESC
                LIMIT 10000
            """
            df = pd.read_sql_query(query, conn)

        if df.empty:
            return df

        # JSON parsing for common columns (if they exist)
        if "metrics" in df.columns:
            df["metrics"] = df["metrics"].apply(lambda x: _normalize_json_like(x, want_list=False))
        if "user_metrics" in df.columns:
            df["user_metrics"] = df["user_metrics"].apply(lambda x: _normalize_json_like(x, want_list=False))
        if "tickers" in df.columns:
            df["tickers"] = df["tickers"].apply(lambda x: _normalize_json_like(x, want_list=True))
        if "hashtags" in df.columns:
            df["hashtags"] = df["hashtags"].apply(lambda x: _normalize_json_like(x, want_list=True))

        # Timestamps
        if "created_at" in df.columns:
            df["created_at"] = _to_utc(df["created_at"])
        if "processed_at" in df.columns:
            df["processed_at"] = _to_utc(df["processed_at"])

        # Ensure we expose a uniform 'sentiment_confidence' column
        if "sentiment_confidence" not in df.columns and "confidence" in df.columns:
            df["sentiment_confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

        # Ensure expected sentiment columns exist (even if NaN) for downstream code
        for col in ("sentiment_label", "sentiment_score", "sentiment_confidence"):
            if col not in df.columns:
                df[col] = pd.Series([None] * len(df))

        return df

    except Exception as e:
        st.error(f"Error loading sentiment data from {db_file} — {e}")
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_stock_prices(symbols: List[str]) -> Dict[str, Dict]:
    """Get current stock prices via StockDataProvider (if available)."""
    if not MODELS_AVAILABLE or not symbols:
        return {}
    try:
        provider = StockDataProvider()
        return provider.get_multiple_current_prices(symbols)
    except Exception as e:
        st.error(f"Error fetching stock prices: {e}")
        return {}

# ──────────────────────────────────────────────────────────────────────────────
# UI helpers (ORIGINAL)
# ──────────────────────────────────────────────────────────────────────────────
def render_sentiment_badge(sentiment_label: str, sentiment_score: float) -> str:
    if sentiment_label == 'positive':
        return f'<span class="sentiment-positive">😊 Positive ({sentiment_score:.2f})</span>'
    elif sentiment_label == 'negative':
        return f'<span class="sentiment-negative">😞 Negative ({sentiment_score:.2f})</span>'
    return f'<span class="sentiment-neutral">😐 Neutral ({sentiment_score:.2f})</span>'

def calculate_sentiment_metrics(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {}
    metrics: Dict = {}
    counts = df["sentiment_label"].value_counts(dropna=True)
    total = len(df)
    metrics["positive_pct"] = counts.get("positive", 0) / total * 100
    metrics["negative_pct"] = counts.get("negative", 0) / total * 100
    metrics["neutral_pct"]  = counts.get("neutral", 0) / total * 100
    metrics["avg_sentiment"] = pd.to_numeric(df["sentiment_score"], errors="coerce").mean()
    metrics["sentiment_volatility"] = pd.to_numeric(df["sentiment_score"], errors="coerce").std()

    metrics["very_positive"] = (pd.to_numeric(df["sentiment_score"], errors="coerce") > 0.7).sum()
    metrics["very_negative"] = (pd.to_numeric(df["sentiment_score"], errors="coerce") < -0.7).sum()

    now = datetime.now(timezone.utc)
    recent = df[df["created_at"] >= now - timedelta(hours=6)]
    prev   = df[(df["created_at"] >= now - timedelta(hours=12)) & (df["created_at"] < now - timedelta(hours=6))]
    if not recent.empty and not prev.empty:
        metrics["sentiment_trend"] = pd.to_numeric(recent["sentiment_score"], errors="coerce").mean() - \
                                     pd.to_numeric(prev["sentiment_score"], errors="coerce").mean()
    else:
        metrics["sentiment_trend"] = 0.0
    return metrics

# ──────────────────────────────────────────────────────────────────────────────
# NEW: Enhanced asset tracking sections
# ──────────────────────────────────────────────────────────────────────────────
def render_asset_tracking_overview(df: pd.DataFrame, config_assets: Dict):
    """Show detailed asset tracking information"""
    st.header("🎯 Asset Tracking Overview")
    st.markdown('<div class="asset-tracker">📡 Monitoring 20 assets across 13 subreddits with AI sentiment analysis</div>', unsafe_allow_html=True)
    
    # Show configured assets
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📈 Stock Tickers")
        tickers_str = " • ".join(config_assets.get('tickers', []))
        st.info(tickers_str)
    
    with col2:
        st.subheader("🏢 Brand Names")
        brands_str = " • ".join(config_assets.get('brands', []))
        st.info(brands_str)
    
    with col3:
        st.subheader("₿ Cryptocurrencies")
        crypto_str = " • ".join(config_assets.get('crypto', []))
        st.info(crypto_str)
    
    # Analyze asset mentions in current data
    if not df.empty:
        st.subheader("📊 Asset Mentions in Current Data")
        
        # Extract mentions from all posts
        all_mentions = []
        asset_post_map = {}
        
        for _, row in df.iterrows():
            text = str(row.get('text', '')) + ' ' + str(row.get('title', ''))
            mentions = extract_asset_mentions(text, config_assets)
            all_mentions.extend(mentions)
            
            # Track which posts mention each asset
            for mention in mentions:
                if mention not in asset_post_map:
                    asset_post_map[mention] = []
                asset_post_map[mention].append({
                    'sentiment_score': row.get('sentiment_score', 0),
                    'sentiment_label': row.get('sentiment_label', 'neutral'),
                    'subreddit': row.get('subreddit', 'unknown'),
                    'created_at': row.get('created_at')
                })
        
        if all_mentions:
            mention_counts = Counter(all_mentions)
            
            # Create asset mention summary
            asset_data = []
            for asset, count in mention_counts.most_common(15):  # Top 15
                posts = asset_post_map[asset]
                avg_sentiment = sum(float(p.get('sentiment_score', 0)) for p in posts) / len(posts)
                
                asset_data.append({
                    'Asset': asset,
                    'Mentions': count,
                    'Avg Sentiment': round(avg_sentiment, 3),
                    'Sentiment Label': 'Positive' if avg_sentiment > 0.1 else 'Negative' if avg_sentiment < -0.1 else 'Neutral',
                    'Top Subreddit': Counter(p['subreddit'] for p in posts).most_common(1)[0][0]
                })
            
            asset_df = pd.DataFrame(asset_data)
            
            # Split into two columns for better display
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(asset_df.head(8), use_container_width=True, hide_index=True)
            
            with col2:
                if len(asset_df) > 8:
                    st.dataframe(asset_df.tail(len(asset_df)-8), use_container_width=True, hide_index=True)
            
            # Asset sentiment visualization
            fig = px.bar(
                asset_df.head(10), 
                x='Asset', 
                y='Mentions', 
                color='Avg Sentiment',
                color_continuous_scale='RdYlGn',
                title="Top 10 Most Mentioned Assets",
                text='Mentions'
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("No specific asset mentions detected. The sentiment analysis is capturing general market discussions.")

def render_collection_details():
    """Show data collection configuration details"""
    st.header("🔍 Data Collection Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📡 Data Sources")
        subreddits = [
            'stocks', 'investing', 'SecurityAnalysis', 'ValueInvesting',
            'technology', 'Apple', 'Microsoft', 'nvidia', 'teslamotors',
            'cryptocurrency', 'Bitcoin', 'ethereum', 'wallstreetbets'
        ]
        
        st.write("**Monitored Subreddits:**")
        subreddit_cols = st.columns(2)
        for i, sub in enumerate(subreddits):
            with subreddit_cols[i % 2]:
                st.write(f"• r/{sub}")
    
    with col2:
        st.subheader("⚙️ Collection Settings")
        st.info("""
        **Search Strategy:**
        • Asset-specific searches across subreddits
        • Combines post titles + content
        • 30-minute collection intervals
        • Max 200 posts per cycle
        • Deduplication by post ID
        
        **AI Analysis:**
        • RoBERTa transformer model
        • Real-time sentiment scoring
        • Confidence tracking
        • Asset mention extraction
        """)

def render_sentiment_by_asset(df: pd.DataFrame, config_assets: Dict):
    """Show sentiment breakdown by specific assets"""
    st.header("📈 Asset-Specific Sentiment Analysis")
    
    if df.empty:
        st.info("No data available for asset-specific analysis")
        return
    
    # Extract asset mentions and create asset-post mapping
    asset_sentiments = {}
    
    for _, row in df.iterrows():
        text = str(row.get('text', '')) + ' ' + str(row.get('title', ''))
        mentions = extract_asset_mentions(text, config_assets)
        
        for asset in mentions:
            if asset not in asset_sentiments:
                asset_sentiments[asset] = {
                    'scores': [],
                    'labels': [],
                    'posts': [],
                    'subreddits': []
                }
            
            score = row.get('sentiment_score', 0)
            # Ensure score is numeric
            try:
                score = float(score) if score is not None else 0.0
            except (ValueError, TypeError):
                score = 0.0
            
            asset_sentiments[asset]['scores'].append(score)
            asset_sentiments[asset]['labels'].append(row.get('sentiment_label', 'neutral'))
            asset_sentiments[asset]['posts'].append(str(row.get('text', ''))[:100] + '...')
            asset_sentiments[asset]['subreddits'].append(row.get('subreddit', 'unknown'))
    
    if not asset_sentiments:
        st.warning("No specific asset mentions found in this timeframe. Try expanding the time range or check if data collection is running.")
        return
    
    # Create tabs for different asset categories
    tab1, tab2, tab3 = st.tabs(["🏢 Stocks", "₿ Crypto", "📊 All Assets"])
    
    with tab1:
        stock_assets = {k: v for k, v in asset_sentiments.items() 
                       if k in config_assets.get('tickers', []) + config_assets.get('brands', [])}
        if stock_assets:
            for asset, data in stock_assets.items():
                if data['scores']:  # Ensure we have scores
                    avg_score = sum(data['scores']) / len(data['scores'])
                    sentiment_emoji = "📈" if avg_score > 0.1 else "📉" if avg_score < -0.1 else "➡️"
                    with st.expander(f"{sentiment_emoji} {asset} - Avg Sentiment: {avg_score:.3f} ({len(data['scores'])} mentions)"):
                        col1, col2 = st.columns(2)
                        with col1:
                            sentiment_dist = Counter(data['labels'])
                            st.write("**Sentiment Distribution:**")
                            for label, count in sentiment_dist.items():
                                emoji = "😊" if label == "positive" else "😞" if label == "negative" else "😐"
                                st.write(f"{emoji} {label.title()}: {count}")
                        with col2:
                            if data['subreddits']:
                                top_subreddit = Counter(data['subreddits']).most_common(1)[0]
                                st.write(f"**Most Active Subreddit:** r/{top_subreddit[0]} ({top_subreddit[1]} posts)")
                            if len(data['scores']) > 1:
                                st.write(f"**Score Range:** {min(data['scores']):.3f} to {max(data['scores']):.3f}")
                            else:
                                st.write(f"**Single Score:** {data['scores'][0]:.3f}")
        else:
            st.info("No stock mentions detected in current timeframe")
    
    with tab2:
        crypto_assets = {k: v for k, v in asset_sentiments.items() 
                        if k.lower() in [c.lower() for c in config_assets.get('crypto', [])]}
        if crypto_assets:
            for asset, data in crypto_assets.items():
                if data['scores']:  # Ensure we have scores
                    avg_score = sum(data['scores']) / len(data['scores'])
                    sentiment_emoji = "🚀" if avg_score > 0.1 else "💥" if avg_score < -0.1 else "🔄"
                    with st.expander(f"{sentiment_emoji} {asset} - Avg Sentiment: {avg_score:.3f} ({len(data['scores'])} mentions)"):
                        col1, col2 = st.columns(2)
                        with col1:
                            sentiment_dist = Counter(data['labels'])
                            st.write("**Sentiment Distribution:**")
                            for label, count in sentiment_dist.items():
                                emoji = "😊" if label == "positive" else "😞" if label == "negative" else "😐"
                                st.write(f"{emoji} {label.title()}: {count}")
                        with col2:
                            if data['subreddits']:
                                top_subreddit = Counter(data['subreddits']).most_common(1)[0]
                                st.write(f"**Most Active Subreddit:** r/{top_subreddit[0]} ({top_subreddit[1]} posts)")
                            if len(data['scores']) > 1:
                                st.write(f"**Score Range:** {min(data['scores']):.3f} to {max(data['scores']):.3f}")
                            else:
                                st.write(f"**Single Score:** {data['scores'][0]:.3f}")
        else:
            st.info("No crypto mentions detected in current timeframe")
    
    with tab3:
        # Summary chart of all assets - FIXED VERSION
        asset_summary = []
        for asset, data in asset_sentiments.items():
            if data['scores']:  # Only process assets with scores
                # Calculate volatility safely
                scores_series = pd.Series(data['scores'])
                volatility = scores_series.std()
                
                # Handle NaN volatility (single data point or all same values)
                if pd.isna(volatility) or volatility == 0:
                    volatility = 0.01  # Small default value
                
                asset_summary.append({
                    'Asset': asset,
                    'Mentions': len(data['scores']),
                    'Avg Sentiment': sum(data['scores']) / len(data['scores']),
                    'Volatility': volatility
                })
        
        if asset_summary:
            summary_df = pd.DataFrame(asset_summary)
            
            # Double-check: ensure no NaN or zero values remain
            summary_df['Volatility'] = summary_df['Volatility'].fillna(0.01)
            summary_df.loc[summary_df['Volatility'] == 0, 'Volatility'] = 0.01
            
            # Create the scatter plot
            try:
                fig = px.scatter(
                    summary_df, 
                    x='Mentions', 
                    y='Avg Sentiment',
                    size='Volatility',
                    color='Asset',
                    title="Asset Sentiment vs Mention Volume",
                    hover_data=['Volatility'],
                    size_max=30,  # Limit max bubble size
                    labels={
                        'Mentions': 'Number of Mentions',
                        'Avg Sentiment': 'Average Sentiment Score',
                        'Volatility': 'Sentiment Volatility'
                    }
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show summary stats
                st.subheader("📊 Asset Summary Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    most_mentioned = summary_df.loc[summary_df['Mentions'].idxmax()]
                    st.metric("Most Mentioned", most_mentioned['Asset'], f"{most_mentioned['Mentions']} posts")
                with col2:
                    most_positive = summary_df.loc[summary_df['Avg Sentiment'].idxmax()]
                    st.metric("Most Positive", most_positive['Asset'], f"{most_positive['Avg Sentiment']:.3f}")
                with col3:
                    most_volatile = summary_df.loc[summary_df['Volatility'].idxmax()]
                    st.metric("Most Volatile", most_volatile['Asset'], f"{most_volatile['Volatility']:.3f}")
                    
            except Exception as e:
                st.error(f"Error creating scatter plot: {e}")
                st.dataframe(summary_df, use_container_width=True)
        else:
            st.info("No asset data available for visualization")
# ──────────────────────────────────────────────────────────────────────────────
# ORIGINAL dashboard sections (preserved exactly)
# ──────────────────────────────────────────────────────────────────────────────
def render_sentiment_overview(df: pd.DataFrame):
    st.header("🤖 AI Sentiment Analysis Overview")
    if df.empty:
        st.warning("No sentiment data available. Run AI collection to generate sentiment scores.")
        return

    m = calculate_sentiment_metrics(df)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Average Sentiment", f"{m.get('avg_sentiment', 0):.3f}",
                  delta=f"{m.get('sentiment_trend', 0):+.3f} (6h trend)")
    with c2:
        st.metric("Positive Posts", f"{m.get('positive_pct', 0):.1f}%",
                  delta=f"{int(m.get('very_positive', 0))} very positive")
    with c3:
        st.metric("Negative Posts", f"{m.get('negative_pct', 0):.1f}%",
                  delta=f"{int(m.get('very_negative', 0))} very negative")
    with c4:
        st.metric("Sentiment Volatility", f"{m.get('sentiment_volatility', 0):.3f}",
                  help="Standard deviation of sentiment scores")

def render_sentiment_charts(df: pd.DataFrame):
    st.header("📊 Sentiment Analytics")
    if df.empty:
        return
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Sentiment Distribution")
        counts = df["sentiment_label"].value_counts()
        colors = {'positive': '#00C851', 'negative': '#FF4444', 'neutral': '#6C757D'}
        fig = px.pie(values=counts.values, names=counts.index, title="Overall Sentiment Distribution",
                     color=counts.index, color_discrete_map=colors)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Sentiment Over Time")
        tmp = df.copy()
        tmp["hour"] = tmp["created_at"].dt.floor("H")
        hourly = tmp.groupby("hour", as_index=False)["sentiment_score"].mean(numeric_only=True)
        line = px.line(hourly, x="hour", y="sentiment_score", title="Average Sentiment Trend",
                       color_discrete_sequence=['#1f77b4'])
        line.add_hline(y=0, line_dash="dash", line_color="gray")
        line.update_layout(yaxis_range=[-1, 1])
        st.plotly_chart(line, use_container_width=True)

def render_stock_correlation_analysis(df: pd.DataFrame):
    st.header("📈 Stock Price Correlation Analysis")
    if not MODELS_AVAILABLE:
        st.info("Install stock add-ons for live prices: `pip install yfinance`")
        return
    if df.empty:
        st.info("No sentiment data available for correlation analysis.")
        return

    # Collect mentioned tickers
    mentioned = set()
    if "tickers" in df.columns:
        for lst in df["tickers"]:
            if isinstance(lst, list):
                mentioned.update(lst)

    main = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL', 'META', 'AMZN']
    tickers = [t for t in main if t in mentioned]
    if not tickers:
        st.info("No major stock tickers found in recent posts.")
        return

    prices = get_stock_prices(tickers)
    rows = []
    for t in tickers:
        mask = df["tickers"].apply(lambda x: t in x if isinstance(x, list) else False)
        tdf = df[mask]
        if not tdf.empty:
            rows.append({
                "ticker": t,
                "sentiment": pd.to_numeric(tdf["sentiment_score"], errors="coerce").mean(),
                "posts": len(tdf),
                "price": prices.get(t, {}).get("current_price", 0.0),
                "change_pct": prices.get(t, {}).get("day_change_percent", 0.0)
            })

    if not rows:
        st.info("No sentiment rows for selected tickers yet.")
        return

    sdf = pd.DataFrame(rows)
    fig = px.scatter(
        sdf, x="sentiment", y="change_pct", size="posts", color="ticker",
        hover_data=["price"], title="Sentiment vs Daily Price Change",
        labels={"sentiment": "Avg Sentiment", "change_pct": "Daily Change (%)", "posts": "Posts"}
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Stocks by Mentions")
    for row in sdf.sort_values("posts", ascending=False).itertuples(index=False):
        s_color = "metric-positive" if row.sentiment > 0 else "metric-negative" if row.sentiment < 0 else ""
        p_color = "metric-positive" if row.change_pct > 0 else "metric-negative" if row.change_pct < 0 else ""
        st.markdown(
            f"**{row.ticker}** — {int(row.posts)} posts  \n"
            f"- Price: ${row.price:.2f} <span class='{p_color}'>({row.change_pct:+.2f}%)</span>  \n"
            f"- Sentiment: <span class='{s_color}'>{row.sentiment:.3f}</span>",
            unsafe_allow_html=True
        )

def render_ai_insights(df: pd.DataFrame):
    st.header("🧠 AI Insights & Alerts")
    if df.empty:
        return

    alerts = []
    pos_ext = pd.to_numeric(df["sentiment_score"], errors="coerce") > 0.7
    neg_ext = pd.to_numeric(df["sentiment_score"], errors="coerce") < -0.7
    if pos_ext.sum() > 0:
        alerts.append(("Extreme Positive Sentiment",
                       f"{int(pos_ext.sum())} posts with very positive sentiment detected",
                       "high" if pos_ext.sum() > 10 else "medium"))
    if neg_ext.sum() > 0:
        alerts.append(("Extreme Negative Sentiment",
                       f"{int(neg_ext.sum())} posts with very negative sentiment detected",
                       "high" if neg_ext.sum() > 10 else "medium"))

    recent_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    vol = (df["created_at"] >= recent_hour).sum()
    if vol > 20:
        alerts.append(("High Volume", f"High posting volume: {vol} posts in last hour", "medium"))

    if alerts:
        for title, desc, sev in alerts:
            emoji = "🔴" if sev == "high" else "🟡"
            st.markdown(f"<div class='alert-banner'>{emoji} {title}: {desc}</div>", unsafe_allow_html=True)
    else:
        st.success("✅ No alerts detected")

    # Quick insights
    st.subheader("📋 Key Insights")
    insights: List[str] = []
    all_ticks: List[str] = []
    if "tickers" in df.columns:
        for lst in df["tickers"]:
            if isinstance(lst, list):
                all_ticks.extend(lst)
    if all_ticks:
        top_t, cnt = Counter(all_ticks).most_common(1)[0]
        insights.append(f"🏆 Most discussed: {top_t} ({cnt} mentions)")

    avg = pd.to_numeric(df["sentiment_score"], errors="coerce").mean()
    if avg > 0.1:
        insights.append(f"📈 Overall bullish sentiment (avg: {avg:.3f})")
    elif avg < -0.1:
        insights.append(f"📉 Overall bearish sentiment (avg: {avg:.3f})")
    else:
        insights.append(f"➡️ Neutral market sentiment (avg: {avg:.3f})")

    last_hour = (df["created_at"] >= datetime.now(timezone.utc) - timedelta(hours=1)).sum()
    if last_hour:
        insights.append(f"🕐 {int(last_hour)} posts in last hour")

    for msg in insights:
        st.info(msg)

def render_top_sentiment_posts(df: pd.DataFrame):
    st.header("🔥 Top Sentiment Posts")
    if df.empty:
        return
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Most Positive Posts")
        top_pos = df.nlargest(5, "sentiment_score")
        for _, row in top_pos.iterrows():
            badge = render_sentiment_badge(str(row.get("sentiment_label", "neutral")),
                                           float(row.get("sentiment_score", 0.0)))
            with st.expander(f"😊 Score: {row.get('sentiment_score', 0):.3f} | r/{row.get('subreddit', 'unknown')}"):
                st.markdown(badge, unsafe_allow_html=True)
                txt = str(row.get("text", ""))[:300]
                st.write("**Text:**", txt + ("..." if len(str(row.get("text",""))) > 300 else ""))
                if isinstance(row.get("tickers", None), list) and row["tickers"]:
                    st.write("**Tickers:**", ", ".join(row["tickers"]))
    with c2:
        st.subheader("Most Negative Posts")
        top_neg = df.nsmallest(5, "sentiment_score")
        for _, row in top_neg.iterrows():
            badge = render_sentiment_badge(str(row.get("sentiment_label", "neutral")),
                                           float(row.get("sentiment_score", 0.0)))
            with st.expander(f"😞 Score: {row.get('sentiment_score', 0):.3f} | r/{row.get('subreddit', 'unknown')}"):
                st.markdown(badge, unsafe_allow_html=True)
                txt = str(row.get("text", ""))[:300]
                st.write("**Text:**", txt + ("..." if len(str(row.get("text",""))) > 300 else ""))
                if isinstance(row.get("tickers", None), list) and row["tickers"]:
                    st.write("**Tickers:**", ", ".join(row["tickers"]))

# ──────────────────────────────────────────────────────────────────────────────
# Main function (ENHANCED)
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.title("🤖 AI-Enhanced Sentiment Dashboard")
    st.caption("Advanced sentiment analysis with transformer/VADER models and stock correlation")

    # Load configuration
    config_assets = load_config_assets()
    
    # Show where we're reading data from (helpful for debugging path issues)
    st.sidebar.markdown("**Database**")
    st.sidebar.code(str(DB_PATH))

    df = load_sentiment_data(DB_PATH)

    if df.empty:
        st.warning(
            "No AI sentiment data found.\n\n"
            "1) Run a collection cycle with AI enabled:\n"
            "   `python \"sentiment analysis/ingestion/orchestrator.py\" --with-ai --mode once`\n"
            "2) Refresh this page."
        )
        return

    # Sidebar filters
    st.sidebar.header("🎛️ AI Dashboard Controls")
    timeframe = st.sidebar.selectbox("Analysis Timeframe",
                                     ["1 Hour", "6 Hours", "24 Hours", "7 Days"], index=2)
    sentiment_filter = st.sidebar.selectbox(
        "Sentiment Filter", ["All Sentiments", "Positive Only", "Negative Only", "Neutral Only"]
    )

    hours = {"1 Hour": 1, "6 Hours": 6, "24 Hours": 24, "7 Days": 168}[timeframe]
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    filtered = df[df["created_at"] >= cutoff].copy()

    if sentiment_filter != "All Sentiments":
        want = {"Positive Only": "positive", "Negative Only": "negative", "Neutral Only": "neutral"}[sentiment_filter]
        filtered = filtered[filtered["sentiment_label"] == want]

    if filtered.empty:
        st.warning(f"No sentiment data found for the selected filters in the past {timeframe.lower()}.")
        return

    # NEW SECTIONS - Enhanced asset tracking
    render_asset_tracking_overview(filtered, config_assets)
    render_collection_details()
    render_sentiment_by_asset(filtered, config_assets)
    
    # ORIGINAL SECTIONS - All preserved
    render_sentiment_overview(filtered)
    render_sentiment_charts(filtered)
    render_stock_correlation_analysis(filtered)
    render_ai_insights(filtered)
    render_top_sentiment_posts(filtered)

    # Enhanced sidebar stats
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Enhanced Tracking Stats")
    total_assets = len(config_assets.get('tickers', [])) + len(config_assets.get('brands', [])) + len(config_assets.get('crypto', []))
    st.sidebar.metric("Total Assets Tracked", total_assets)
    st.sidebar.metric("Posts with AI Sentiment", len(filtered))
    st.sidebar.metric("Timeframe", timeframe)
    
    # Show asset breakdown
    st.sidebar.write("**Asset Categories:**")
    st.sidebar.write(f"• Stocks: {len(config_assets.get('tickers', []))}")
    st.sidebar.write(f"• Brands: {len(config_assets.get('brands', []))}")
    st.sidebar.write(f"• Crypto: {len(config_assets.get('crypto', []))}")
    
    if MODELS_AVAILABLE:
        st.sidebar.success("🤖 AI Extras Available")
    else:
        st.sidebar.error("❌ AI Extras Missing (optional)")

    latest = pd.to_datetime(filtered["processed_at"], utc=True, errors="coerce")
    if latest.notna().any():
        st.sidebar.caption(f"Last AI analysis: {latest.max().strftime('%H:%M:%S UTC')}")

if __name__ == "__main__":
    main()
