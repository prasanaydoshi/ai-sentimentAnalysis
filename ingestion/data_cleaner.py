# ingestion/data_cleaner.py
import re
import emoji
import pandas as pd
from typing import List, Dict
import logging

# Fix langdetect import
try:
    from langdetect import detect
    from langdetect.lang_detect_exception import LangDetectError
except ImportError:
    # Fallback if langdetect not available
    def detect(text):
        return 'en'
    class LangDetectError(Exception):
        pass

logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Text preprocessing and cleaning for social media content.
    Design choice: Aggressive cleaning while preserving sentiment-bearing content.
    """
    
    def __init__(self):
        # Regex patterns for cleaning
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.cashtag_pattern = re.compile(r'\$[A-Z]{1,5}\b')
        self.whitespace_pattern = re.compile(r'\s+')
        
    def clean_text(self, text: str, preserve_tickers: bool = True) -> str:
        """
        Clean social media text while preserving sentiment.
        
        Design choices:
        - Remove URLs (no sentiment value, add noise)
        - Remove @mentions (privacy, reduced noise)  
        - Keep hashtags and cashtags (sentiment indicators)
        - Convert emojis to text (preserve emotional context)
        - Normalize whitespace
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert emojis to text descriptions (preserves sentiment)
        text = emoji.demojize(text, delimiters=(" ", " "))
        
        # Remove URLs
        text = self.url_pattern.sub('', text)
        
        # Remove @mentions (but preserve the content structure)
        text = self.mention_pattern.sub('', text)
        
        # Keep hashtags and cashtags if preserve_tickers=True
        if not preserve_tickers:
            text = self.hashtag_pattern.sub('', text)
            text = self.cashtag_pattern.sub('', text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Strip and return
        return text.strip()
    
    def detect_language(self, text: str) -> str:
        """Detect language of text content."""
        try:
            if len(text.strip()) < 3:
                return 'unknown'
            return detect(text)
        except (LangDetectError, Exception):
            # Return 'en' as default if detection fails
            return 'en'
    
    def extract_tickers(self, text: str) -> List[str]:
        """Extract cashtags/tickers from text."""
        matches = self.cashtag_pattern.findall(text.upper())
        return [match.replace('$', '') for match in matches]
    
    def extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text."""
        matches = self.hashtag_pattern.findall(text.lower())
        return [match.replace('#', '') for match in matches]
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process entire DataFrame of social media posts.
        Design choice: Add derived features during cleaning to avoid recomputation.
        """
        if df.empty:
            return df
        
        logger.info(f"Processing {len(df)} posts...")
        
        # Create cleaned text column
        df['cleaned_text'] = df['text'].apply(lambda x: self.clean_text(x, preserve_tickers=True))
        
        # Add language detection
        df['detected_language'] = df['cleaned_text'].apply(self.detect_language)
        
        # Extract features
        df['extracted_tickers'] = df['text'].apply(self.extract_tickers)
        df['extracted_hashtags'] = df['text'].apply(self.extract_hashtags)
        
        # Filter out very short posts (likely low quality)
        df = df[df['cleaned_text'].str.len() >= 10]
        
        # Filter for English content (for MVP)
        df = df[df['detected_language'] == 'en']
        
        logger.info(f"After cleaning: {len(df)} posts remain")
        return df
