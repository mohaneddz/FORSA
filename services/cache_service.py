import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

CACHE_FOLDER = Path("cache")
CACHE_FOLDER.mkdir(exist_ok=True)

class CacheService:
    def __init__(self):
        self.cache_folder = CACHE_FOLDER
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def _get_cache_key(self, query: str) -> str:
        """Generate a unique cache key from the query."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache entry."""
        return self.cache_folder / f"{cache_key}.json"
    
    def get_from_cache(self, query: str) -> Optional[tuple[List[Dict[str, Any]], str, float]]:
        """
        Retrieve cached results for a query.
        Returns tuple of (results, original_query, timestamp) if found, None if cache miss.
        """
        start_time = time.time()
        try:
            cache_key = self._get_cache_key(query)
            cache_path = self._get_cache_path(cache_key)
            
            if not cache_path.exists():
                elapsed = time.time() - start_time
                logger.info(f"⚡ Cache lookup: MISS | Time: {elapsed:.4f}s | Query: {query[:50]}...")
                return None
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            elapsed = time.time() - start_time
            original_query = cache_data.get('query', query)
            cache_timestamp = cache_data.get('timestamp', 0)
            
            logger.info(f"⚡ Cache lookup: HIT ✓ | Time: {elapsed:.4f}s | Original query: '{original_query}' | Cached: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cache_timestamp))}")
            
            return (cache_data.get('results', []), original_query, cache_timestamp)
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"⚡ Cache lookup: ERROR | Time: {elapsed:.4f}s | Error: {e}")
            return None
    
    def save_to_cache(self, query: str, results: List[Dict[str, Any]]):
        """
        Save retrieval results to cache.
        """
        try:
            cache_key = self._get_cache_key(query)
            cache_path = self._get_cache_path(cache_key)
            
            cache_data = {
                'query': query,
                'results': results,
                'timestamp': time.time()
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved to cache: {query[:50]}...")
            
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    async def get_results_with_cache(
        self, 
        query: str, 
        db_retrieval_func,
        *args,
        **kwargs
    ) -> tuple[List[Dict[str, Any]], str, Dict[str, float]]:
        """
        Retrieve results using parallel cache and database lookup.
        Returns whichever finishes first.
        
        Returns:
            tuple: (results, source, timing_info) where source is either 'cache' or 'database'
                   timing_info contains 'cache_time', 'db_time', 'total_time'
        """
        overall_start = time.time()
        cache_start = time.time()
        db_start = time.time()
        
        loop = asyncio.get_event_loop()
        
        timing_info = {'cache_time': 0.0, 'db_time': 0.0, 'total_time': 0.0}
        
        # Create tasks for both cache and database retrieval
        cache_task = loop.run_in_executor(
            self.executor, 
            self.get_from_cache, 
            query
        )
        
        db_task = loop.run_in_executor(
            self.executor,
            db_retrieval_func,
            query,
            *args
        )
        
        logger.info(f"🔄 Starting parallel retrieval: cache + database for query: '{query[:50]}...'")
        
        # Wait for the first one to complete
        done, pending = await asyncio.wait(
            [cache_task, db_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Get the result from the first completed task
        first_result = None
        source = None
        original_query = None
        
        for task in done:
            if task == cache_task:
                cache_result = await task
                timing_info['cache_time'] = time.time() - cache_start
                
                if cache_result is not None:
                    results, original_query, cache_timestamp = cache_result
                    first_result = results
                    source = 'cache'
                    logger.info(f"✅ Using CACHE result (finished first) | Cache time: {timing_info['cache_time']:.4f}s")
                    # Cancel pending database task
                    for t in pending:
                        t.cancel()
                        logger.info("🚫 Database retrieval cancelled (cache hit)")
                    break
            elif task == db_task:
                db_result = await task
                timing_info['db_time'] = time.time() - db_start
                first_result = db_result
                source = 'database'
                logger.info(f"✅ Using DATABASE result | DB time: {timing_info['db_time']:.4f}s")
                # Save to cache for future use
                if db_result:
                    loop.run_in_executor(
                        self.executor,
                        self.save_to_cache,
                        query,
                        db_result
                    )
                    logger.info("💾 Saving database result to cache for future use")
                break
        
        # If cache was empty but finished first, wait for database
        if first_result is None:
            logger.info("⏳ Cache was empty, waiting for database result...")
            for task in pending:
                db_result = await task
                timing_info['db_time'] = time.time() - db_start
                first_result = db_result
                source = 'database'
                logger.info(f"✅ Database result received | DB time: {timing_info['db_time']:.4f}s")
                # Save to cache
                if db_result:
                    loop.run_in_executor(
                        self.executor,
                        self.save_to_cache,
                        query,
                        db_result
                    )
        
        timing_info['total_time'] = time.time() - overall_start
        logger.info(f"📊 Parallel retrieval complete | Source: {source.upper()} | Total time: {timing_info['total_time']:.4f}s")
        
        return first_result or [], source, timing_info
    
    def clear_cache(self):
        """Clear all cache files."""
        try:
            for cache_file in self.cache_folder.glob("*.json"):
                cache_file.unlink()
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        try:
            cache_files = list(self.cache_folder.glob("*.json"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                'total_entries': len(cache_files),
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2)
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}

cache_service = CacheService()
