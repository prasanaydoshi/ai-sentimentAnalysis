# models/stock_data.py
"""
Stock price data integration for sentiment correlation analysis.
Uses Yahoo Finance API for real-time and historical stock data.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import requests
import yfinance as yf
import time

logger = logging.getLogger(__name__)

class StockDataProvider:
    """
    Professional stock data provider with multiple data sources and caching.
    """
    
    def __init__(self):
        """Initialize stock data provider."""
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
        
        # Common ticker symbol mappings
        self.symbol_mapping = {
            'Apple': 'AAPL',
            'Microsoft': 'MSFT', 
            'Google': 'GOOGL',
            'Amazon': 'AMZN',
            'Tesla': 'TSLA',
            'Meta': 'META',
            'Nvidia': 'NVDA',
            'bitcoin': 'BTC-USD',
            'ethereum': 'ETH-USD',
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD'
        }
        
        logger.info("Stock data provider initialized")
    
    def normalize_symbol(self, symbol: str) -> str:
        """Convert brand names and crypto to proper ticker symbols."""
        if not symbol:
            return symbol
            
        # Check direct mapping
        normalized = self.symbol_mapping.get(symbol, symbol)
        
        # Ensure proper format for crypto
        if symbol.upper() in ['BTC', 'ETH'] and not normalized.endswith('-USD'):
            normalized = f"{symbol.upper()}-USD"
        
        return normalized.upper()
    
    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get current stock price and basic metrics.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with current price data
        """
        symbol = self.normalize_symbol(symbol)
        
        # Check cache first
        cache_key = f"current_{symbol}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current price data
            current_data = {
                'symbol': symbol,
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'previous_close': info.get('previousClose', 0),
                'day_change': 0,
                'day_change_percent': 0,
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Calculate day change
            if current_data['current_price'] and current_data['previous_close']:
                current_data['day_change'] = current_data['current_price'] - current_data['previous_close']
                current_data['day_change_percent'] = (current_data['day_change'] / current_data['previous_close']) * 100
            
            # Cache the result
            self._cache_data(cache_key, current_data)
            
            logger.info(f"Retrieved current price for {symbol}: ${current_data['current_price']:.2f}")
            return current_data
            
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, period: str = "1mo") -> Optional[pd.DataFrame]:
        """
        Get historical stock price data.
        
        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame with historical price data
        """
        symbol = self.normalize_symbol(symbol)
        
        # Check cache
        cache_key = f"historical_{symbol}_{period}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period=period)
            
            if hist_data.empty:
                logger.warning(f"No historical data found for {symbol}")
                return None
            
            # Clean and prepare data
            hist_data = hist_data.reset_index()
            hist_data['Date'] = pd.to_datetime(hist_data['Date'])
            hist_data['symbol'] = symbol
            
            # Calculate additional metrics
            hist_data['daily_return'] = hist_data['Close'].pct_change()
            hist_data['volatility'] = hist_data['daily_return'].rolling(window=5).std()
            
            # Cache the result
            self._cache_data(cache_key, hist_data)
            
            logger.info(f"Retrieved {len(hist_data)} days of historical data for {symbol}")
            return hist_data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    def get_intraday_data(self, symbol: str, interval: str = "1h") -> Optional[pd.DataFrame]:
        """
        Get intraday price data for correlation with sentiment.
        
        Args:
            symbol: Stock ticker symbol  
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with intraday price data
        """
        symbol = self.normalize_symbol(symbol)
        
        try:
            ticker = yf.Ticker(symbol)
            # Get last 7 days of intraday data
            intraday_data = ticker.history(period="7d", interval=interval)
            
            if intraday_data.empty:
                logger.warning(f"No intraday data found for {symbol}")
                return None
            
            # Clean and prepare data
            intraday_data = intraday_data.reset_index()
            intraday_data['Datetime'] = pd.to_datetime(intraday_data['Datetime'])
            intraday_data['symbol'] = symbol
            
            # Calculate price change metrics
            intraday_data['price_change'] = intraday_data['Close'] - intraday_data['Open']
            intraday_data['price_change_percent'] = (intraday_data['price_change'] / intraday_data['Open']) * 100
            
            logger.info(f"Retrieved {len(intraday_data)} intraday data points for {symbol}")
            return intraday_data
            
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return None
    
    def get_multiple_current_prices(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Get current prices for multiple symbols efficiently.
        
        Args:
            symbols: List of ticker symbols
            
        Returns:
            Dictionary mapping symbols to price data
        """
        results = {}
        
        for symbol in symbols:
            try:
                price_data = self.get_current_price(symbol)
                if price_data:
                    results[symbol] = price_data
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching price for {symbol}: {e}")
        
        logger.info(f"Retrieved prices for {len(results)}/{len(symbols)} symbols")
        return results
    
    def _is_cached(self, key: str) -> bool:
        """Check if data is cached and still valid."""
        if key not in self.cache:
            return False
        
        cache_time = self.cache[key]['timestamp']
        return (datetime.now() - cache_time).total_seconds() < self.cache_duration
    
    def _cache_data(self, key: str, data):
        """Cache data with timestamp."""
        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }


class SentimentPriceCorrelator:
    """
    Analyze correlation between sentiment and stock price movements.
    """
    
    def __init__(self, stock_provider: StockDataProvider = None):
        """Initialize correlator with stock data provider."""
        self.stock_provider = stock_provider or StockDataProvider()
        
    def analyze_sentiment_price_correlation(self, sentiment_df: pd.DataFrame, 
                                          symbol: str, 
                                          timeframe_hours: int = 24) -> Dict[str, float]:
        """
        Analyze correlation between sentiment and price movements.
        
        Args:
            sentiment_df: DataFrame with sentiment data
            symbol: Stock symbol to analyze
            timeframe_hours: Time window for analysis
            
        Returns:
            Dictionary with correlation metrics
        """
        try:
            # Get stock price data
            intraday_data = self.stock_provider.get_intraday_data(symbol, interval="1h")
            
            if intraday_data is None or sentiment_df.empty:
                return self._empty_correlation_result()
            
            # Filter sentiment data for the symbol
            symbol_sentiment = self._filter_sentiment_for_symbol(sentiment_df, symbol)
            
            if symbol_sentiment.empty:
                return self._empty_correlation_result()
            
            # Aggregate sentiment by hour
            hourly_sentiment = self._aggregate_sentiment_by_hour(symbol_sentiment)
            
            # Merge with price data
            merged_data = self._merge_sentiment_price_data(hourly_sentiment, intraday_data)
            
            if merged_data.empty or len(merged_data) < 3:
                return self._empty_correlation_result()
            
            # Calculate correlations
            correlation_metrics = self._calculate_correlations(merged_data)
            
            logger.info(f"Calculated sentiment-price correlation for {symbol}")
            return correlation_metrics
            
        except Exception as e:
            logger.error(f"Error calculating correlation for {symbol}: {e}")
            return self._empty_correlation_result()
    
    def _filter_sentiment_for_symbol(self, sentiment_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Filter sentiment data for specific symbol."""
        # Look for symbol in tickers, hashtags, or text
        mask = (
            sentiment_df['tickers'].apply(lambda x: symbol in x if isinstance(x, list) else False) |
            sentiment_df['hashtags'].apply(lambda x: symbol.lower() in x if isinstance(x, list) else False) |
            sentiment_df['text'].str.contains(symbol, case=False, na=False)
        )
        
        return sentiment_df[mask].copy()
    
    def _aggregate_sentiment_by_hour(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sentiment scores by hour."""
        sentiment_df['hour'] = sentiment_df['created_at'].dt.floor('H')
        
        hourly_agg = sentiment_df.groupby('hour').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'positive_prob': 'mean',
            'negative_prob': 'mean',
            'neutral_prob': 'mean'
        }).reset_index()
        
        # Flatten column names
        hourly_agg.columns = ['hour', 'avg_sentiment', 'sentiment_std', 'post_count', 
                             'avg_positive_prob', 'avg_negative_prob', 'avg_neutral_prob']
        
        return hourly_agg
    
    def _merge_sentiment_price_data(self, sentiment_df: pd.DataFrame, 
                                   price_df: pd.DataFrame) -> pd.DataFrame:
        """Merge sentiment and price data by hour."""
        # Ensure datetime columns
        sentiment_df['hour'] = pd.to_datetime(sentiment_df['hour'])
        price_df['hour'] = pd.to_datetime(price_df['Datetime']).dt.floor('H')
        
        # Merge on hour
        merged = pd.merge(sentiment_df, price_df, on='hour', how='inner')
        
        return merged
    
    def _calculate_correlations(self, merged_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate various correlation metrics."""
        correlations = {}
        
        # Basic price-sentiment correlation
        correlations['sentiment_price_corr'] = merged_df['avg_sentiment'].corr(merged_df['Close'])
        correlations['sentiment_volume_corr'] = merged_df['avg_sentiment'].corr(merged_df['Volume'])
        correlations['sentiment_volatility_corr'] = merged_df['avg_sentiment'].corr(merged_df['price_change_percent'])
        
        # Positive/negative sentiment correlations
        correlations['positive_price_corr'] = merged_df['avg_positive_prob'].corr(merged_df['price_change_percent'])
        correlations['negative_price_corr'] = merged_df['avg_negative_prob'].corr(merged_df['price_change_percent'])
        
        # Volume correlations
        correlations['posts_volume_corr'] = merged_df['post_count'].corr(merged_df['Volume'])
        
        # Lead-lag analysis (sentiment leading price)
        if len(merged_df) > 1:
            merged_df_sorted = merged_df.sort_values('hour')
            merged_df_sorted['next_price_change'] = merged_df_sorted['price_change_percent'].shift(-1)
            correlations['sentiment_leads_price'] = merged_df_sorted['avg_sentiment'].corr(
                merged_df_sorted['next_price_change']
            )
        
        # Clean up NaN values
        correlations = {k: v if not pd.isna(v) else 0.0 for k, v in correlations.items()}
        
        return correlations
    
    def _empty_correlation_result(self) -> Dict[str, float]:
        """Return empty correlation result."""
        return {
            'sentiment_price_corr': 0.0,
            'sentiment_volume_corr': 0.0,
            'sentiment_volatility_corr': 0.0,
            'positive_price_corr': 0.0,
            'negative_price_corr': 0.0,
            'posts_volume_corr': 0.0,
            'sentiment_leads_price': 0.0
        }


def test_stock_data_provider():
    """Test the stock data provider."""
    print("🧪 Testing Stock Data Provider...")
    
    provider = StockDataProvider()
    
    # Test symbols
    test_symbols = ['AAPL', 'TSLA', 'BTC-USD']
    
    for symbol in test_symbols:
        print(f"\n--- Testing {symbol} ---")
        
        # Current price
        current = provider.get_current_price(symbol)
        if current:
            print(f"Current Price: ${current['current_price']:.2f}")
            print(f"Day Change: {current['day_change_percent']:.2f}%")
        
        # Historical data
        historical = provider.get_historical_data(symbol, period="5d")
        if historical is not None:
            print(f"Historical data: {len(historical)} days")
    
    print("\n✅ Stock data provider test complete!")


if __name__ == "__main__":
    test_stock_data_provider()
