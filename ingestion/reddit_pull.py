# ingestion/reddit_pull.py
import praw
import pandas as pd
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional
import time

logger = logging.getLogger(__name__)

class RedditIngestion:
    """
    Reddit data ingestion using PRAW.
    Searches relevant subreddits for mentions of target assets.
    """
    
    def __init__(self, config: Dict):
        """Initialize with configuration."""
        self.config = config
        self.reddit = self._setup_client()
        
        # Design choice: Focus on finance and tech subreddits for relevant discussions
        self.target_subreddits = [
            'stocks', 'investing', 'SecurityAnalysis', 'ValueInvesting',
            'technology', 'Apple', 'Microsoft', 'nvidia', 'teslamotors',
            'cryptocurrency', 'Bitcoin', 'ethereum', 'wallstreetbets'
        ]
    
    def _setup_client(self) -> praw.Reddit:
        """Initialize Reddit API client."""
        try:
            reddit = praw.Reddit(
                client_id=self.config['reddit']['client_id'],
                client_secret=self.config['reddit']['client_secret'],
                user_agent=self.config['reddit']['user_agent']
            )
            
            # Test connection
            reddit.user.me()
            logger.info("Reddit API connection successful")
            return reddit
            
        except Exception as e:
            logger.error(f"Failed to connect to Reddit API: {e}")
            raise
    
    def search_subreddit_posts(self, subreddit_name: str, query: str, 
                              limit: int = 25, time_filter: str = 'day') -> List[Dict]:
        """
        Search for posts in a specific subreddit.
        Design choice: Use search rather than streaming for better content quality.
        """
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = []
            
            # Search both hot posts and by query
            search_results = subreddit.search(query, sort='hot', time_filter=time_filter, limit=limit)
            
            for post in search_results:
                processed_post = self._process_reddit_post(post, subreddit_name)
                if processed_post:
                    posts.append(processed_post)
            
            logger.info(f"Fetched {len(posts)} posts from r/{subreddit_name}")
            return posts
            
        except Exception as e:
            logger.error(f"Error fetching from r/{subreddit_name}: {e}")
            return []
    
    def _process_reddit_post(self, post, subreddit_name: str) -> Optional[Dict]:
        """
        Process Reddit post into standardized format.
        Design choice: Include both title and body for comprehensive sentiment analysis.
        """
        try:
            # Combine title and selftext for full context
            full_text = post.title
            if hasattr(post, 'selftext') and post.selftext:
                full_text += " " + post.selftext
            
            processed = {
                'id': post.id,
                'text': full_text,
                'title': post.title,
                'created_at': datetime.fromtimestamp(post.created_utc, tz=timezone.utc).isoformat(),
                'author': str(post.author) if post.author else '[deleted]',
                'subreddit': subreddit_name,
                'source': 'reddit',
                'url': f"https://reddit.com{post.permalink}",
                'metrics': {
                    'score': post.score,
                    'upvote_ratio': post.upvote_ratio,
                    'num_comments': post.num_comments
                },
                'raw_data': {
                    'permalink': post.permalink,
                    'is_self': post.is_self,
                    'over_18': post.over_18
                },
                'processed_at': datetime.now(timezone.utc).isoformat()
            }
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing Reddit post {post.id}: {e}")
            return None
    
    def run_collection_cycle(self, assets: List[str]) -> pd.DataFrame:
        """
        Run complete Reddit collection cycle across all target subreddits.
        Design choice: Parallel collection across subreddits for efficiency.
        """
        logger.info(f"Starting Reddit collection cycle for assets: {assets}")
        
        all_posts = []
        max_posts_per_sub = self.config['settings']['max_reddit_posts'] // len(self.target_subreddits)
        
        for asset in assets:
            for subreddit_name in self.target_subreddits:
                posts = self.search_subreddit_posts(
                    subreddit_name, 
                    asset, 
                    limit=max_posts_per_sub
                )
                all_posts.extend(posts)
                
                # Rate limiting - Reddit API is stricter
                time.sleep(1)
        
        if all_posts:
            df = pd.DataFrame(all_posts)
            # Remove duplicates based on post ID
            df = df.drop_duplicates(subset=['id'])
            logger.info(f"Reddit collection cycle complete: {len(df)} unique posts collected")
            return df
        else:
            logger.warning("No Reddit posts collected in this cycle")
            return pd.DataFrame()