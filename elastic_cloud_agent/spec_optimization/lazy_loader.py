"""
Lazy loading system with caching for OpenAPI specification optimisation.
"""

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models import BaseLanguageModel

from elastic_cloud_agent.spec_optimization.spec_registry import SpecRegistry


class SpecCacheManager:
    """Manages caching and lazy loading of optimised OpenAPI specifications."""

    def __init__(
        self,
        base_spec: Dict[str, Union[str, Dict]],
        llm: BaseLanguageModel,
        cache_dir: Optional[Path] = None,
        max_cache_size: int = 100,
        cache_ttl_seconds: int = 3600,
    ):
        """
        Initialise the spec cache manager.

        Args:
            base_spec: The complete OpenAPI specification
            llm: The language model to use for intent classification
            cache_dir: Directory for persistent cache storage
            max_cache_size: Maximum number of cached specs to keep in memory
            cache_ttl_seconds: Time-to-live for cached specs in seconds
        """
        self.base_spec = base_spec
        self.spec_registry = SpecRegistry(base_spec, llm)
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.cache_ttl_seconds = cache_ttl_seconds

        # In-memory cache
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_access_count: Dict[str, int] = {}
        self._cache_lock = threading.RLock()

        # Background preloading
        self._preloading_enabled = True
        self._preload_thread: Optional[threading.Thread] = None

        # Cache statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "preloads": 0,
        }

        # Ensure cache directory exists
        if self.cache_dir:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                # If directory creation fails, disable persistent caching
                self.cache_dir = None

    def get_spec_lazy(
        self,
        query: str,
        use_persistent_cache: bool = True,
        force_refresh: bool = False,
    ) -> Dict[str, Union[str, Dict]]:
        """
        Get an optimised spec using lazy loading with caching.

        Args:
            query: User query to determine intent and optimisation strategy
            use_persistent_cache: Whether to use disk-based persistent cache
            force_refresh: Force refresh of cached spec

        Returns:
            Optimised OpenAPI specification
        """
        # Generate cache key based on query and base spec hash
        cache_key = self._generate_cache_key(query)

        with self._cache_lock:
            # Check memory cache first
            if not force_refresh and cache_key in self._memory_cache:
                if self._is_cache_valid(cache_key):
                    self._stats["hits"] += 1
                    self._cache_access_count[cache_key] = (
                        self._cache_access_count.get(cache_key, 0) + 1
                    )
                    return self._memory_cache[cache_key]["spec"]

            # Check persistent cache if enabled
            if use_persistent_cache and not force_refresh:
                persistent_spec = self._load_from_persistent_cache(cache_key)
                if persistent_spec:
                    self._stats["hits"] += 1
                    self._store_in_memory_cache(cache_key, persistent_spec)
                    return persistent_spec

            # Cache miss - generate spec
            self._stats["misses"] += 1
            spec = self.spec_registry.get_spec_for_query(query, use_cache=False)

            # Store in caches
            self._store_in_memory_cache(cache_key, spec)
            if use_persistent_cache:
                self._store_in_persistent_cache(cache_key, spec, query)

            return spec

    def preload_common_specs(self, queries: Optional[List[str]] = None) -> None:
        """
        Preload commonly used specifications in the background.

        Args:
            queries: List of common queries to preload. If None, uses default set.
        """
        if not self._preloading_enabled:
            return

        if queries is None:
            queries = [
                "list deployments",
                "create deployment",
                "get deployment status",
                "check cluster health",
                "view billing costs",
                "manage API keys",
                "monitor deployment performance",
            ]

        def preload_worker():
            for query in queries:
                try:
                    if not self._is_query_cached(query):
                        self.get_spec_lazy(query, use_persistent_cache=True)
                        self._stats["preloads"] += 1
                        # Small delay to avoid overwhelming the system
                        time.sleep(0.1)
                except Exception:
                    # Silently continue on preload errors
                    continue

        if self._preload_thread is None or not self._preload_thread.is_alive():
            self._preload_thread = threading.Thread(target=preload_worker, daemon=True)
            self._preload_thread.start()

    def get_cached_queries(self) -> List[str]:
        """
        Get list of queries that are currently cached.

        Returns:
            List of cached query patterns
        """
        with self._cache_lock:
            cached_queries = []
            for cache_key in self._memory_cache:
                if self._is_cache_valid(cache_key):
                    # Try to extract original query from metadata
                    metadata = self._memory_cache[cache_key].get("metadata", {})
                    original_query = metadata.get("original_query", f"key:{cache_key[:8]}")
                    cached_queries.append(original_query)
            return cached_queries

    def invalidate_cache(self, pattern: Optional[str] = None) -> int:
        """
        Invalidate cached specs.

        Args:
            pattern: Optional pattern to match against cache keys. If None, clears all.

        Returns:
            Number of cache entries invalidated
        """
        with self._cache_lock:
            if pattern is None:
                count = len(self._memory_cache)
                self._memory_cache.clear()
                self._cache_timestamps.clear()
                self._cache_access_count.clear()

                # Clear persistent cache
                if self.cache_dir and self.cache_dir.exists():
                    for cache_file in self.cache_dir.glob("*.json"):
                        cache_file.unlink()

                return count

            # Pattern-based invalidation
            keys_to_remove = [key for key in self._memory_cache if pattern in key]

            for key in keys_to_remove:
                del self._memory_cache[key]
                self._cache_timestamps.pop(key, None)
                self._cache_access_count.pop(key, None)

                # Remove from persistent cache
                self._remove_from_persistent_cache(key)

            return len(keys_to_remove)

    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary with cache performance metrics
        """
        with self._cache_lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0

            return {
                "memory_cache_size": len(self._memory_cache),
                "persistent_cache_size": self._get_persistent_cache_size(),
                "hit_rate": hit_rate,
                "total_hits": self._stats["hits"],
                "total_misses": self._stats["misses"],
                "total_evictions": self._stats["evictions"],
                "total_preloads": self._stats["preloads"],
                "most_accessed_specs": self._get_most_accessed_specs(),
                "cache_age_distribution": self._get_cache_age_distribution(),
            }

    def cleanup_expired_cache(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        with self._cache_lock:
            current_time = time.time()
            expired_keys = [
                key
                for key, timestamp in self._cache_timestamps.items()
                if current_time - timestamp > self.cache_ttl_seconds
            ]

            for key in expired_keys:
                del self._memory_cache[key]
                del self._cache_timestamps[key]
                self._cache_access_count.pop(key, None)
                self._remove_from_persistent_cache(key)

            return len(expired_keys)

    def _generate_cache_key(self, query: str) -> str:
        """Generate a cache key for a query."""
        # Include base spec hash to invalidate cache when spec changes
        spec_hash = hashlib.md5(json.dumps(self.base_spec, sort_keys=True).encode()).hexdigest()[:8]

        query_hash = hashlib.md5(query.encode()).hexdigest()[:16]
        return f"spec_{spec_hash}_{query_hash}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if a cache entry is still valid."""
        if cache_key not in self._cache_timestamps:
            return False

        age = time.time() - self._cache_timestamps[cache_key]
        return age < self.cache_ttl_seconds

    def _is_query_cached(self, query: str) -> bool:
        """Check if a query is already cached."""
        cache_key = self._generate_cache_key(query)
        return cache_key in self._memory_cache and self._is_cache_valid(cache_key)

    def _store_in_memory_cache(self, cache_key: str, spec: Dict[str, Union[str, Dict]]) -> None:
        """Store spec in memory cache with LRU eviction."""
        # Evict least recently used items if cache is full
        if len(self._memory_cache) >= self.max_cache_size:
            self._evict_lru_items()

        current_time = time.time()
        self._memory_cache[cache_key] = {
            "spec": spec,
            "metadata": {
                "created_at": current_time,
                "size_bytes": len(json.dumps(spec).encode()),
            },
        }
        self._cache_timestamps[cache_key] = current_time
        self._cache_access_count[cache_key] = 1

    def _evict_lru_items(self) -> None:
        """Evict least recently used cache items."""
        if not self._cache_access_count:
            return

        # Sort by access count (ascending) and then by timestamp (ascending)
        lru_keys = sorted(
            self._cache_access_count.keys(),
            key=lambda k: (self._cache_access_count[k], self._cache_timestamps.get(k, 0)),
        )

        # Remove oldest 25% of cache
        items_to_remove = max(1, len(lru_keys) // 4)
        for key in lru_keys[:items_to_remove]:
            del self._memory_cache[key]
            del self._cache_timestamps[key]
            del self._cache_access_count[key]
            self._stats["evictions"] += 1

    def _load_from_persistent_cache(self, cache_key: str) -> Optional[Dict[str, Union[str, Dict]]]:
        """Load spec from persistent cache."""
        if not self.cache_dir:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None

        try:
            # Check file age
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age > self.cache_ttl_seconds:
                cache_file.unlink()
                return None

            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                # Return just the spec part if it's wrapped in metadata
                if isinstance(cached_data, dict) and "spec" in cached_data:
                    return cached_data["spec"]
                return cached_data
        except (json.JSONDecodeError, OSError):
            # Remove corrupted cache file
            if cache_file.exists():
                cache_file.unlink()
            return None

    def _store_in_persistent_cache(
        self, cache_key: str, spec: Dict[str, Union[str, Dict]], original_query: str
    ) -> None:
        """Store spec in persistent cache."""
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            # Store with metadata
            cache_data = {
                "spec": spec,
                "metadata": {
                    "original_query": original_query,
                    "cached_at": time.time(),
                    "cache_key": cache_key,
                },
            }

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, separators=(",", ":"))
        except OSError:
            # Silently fail on storage errors
            pass

    def _remove_from_persistent_cache(self, cache_key: str) -> None:
        """Remove spec from persistent cache."""
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                cache_file.unlink()
            except OSError:
                pass

    def _get_persistent_cache_size(self) -> int:
        """Get number of files in persistent cache."""
        if not self.cache_dir or not self.cache_dir.exists():
            return 0
        try:
            return len(list(self.cache_dir.glob("*.json")))
        except OSError:
            return 0

    def _get_most_accessed_specs(self) -> List[Dict[str, Any]]:
        """Get the most frequently accessed specs."""
        if not self._cache_access_count:
            return []

        sorted_specs = sorted(self._cache_access_count.items(), key=lambda x: x[1], reverse=True)

        result = []
        for cache_key, access_count in sorted_specs[:5]:
            metadata = self._memory_cache.get(cache_key, {}).get("metadata", {})
            result.append(
                {
                    "cache_key": cache_key[:16] + "...",
                    "access_count": access_count,
                    "size_bytes": metadata.get("size_bytes", 0),
                    "original_query": metadata.get("original_query", "unknown"),
                }
            )

        return result

    def _get_cache_age_distribution(self) -> Dict[str, int]:
        """Get distribution of cache entry ages."""
        if not self._cache_timestamps:
            return {}

        current_time = time.time()
        age_buckets = {
            "under_1min": 0,
            "1_to_5min": 0,
            "5_to_15min": 0,
            "15_to_60min": 0,
            "over_1hour": 0,
        }

        for timestamp in self._cache_timestamps.values():
            age_seconds = current_time - timestamp
            age_minutes = age_seconds / 60

            if age_minutes < 1:
                age_buckets["under_1min"] += 1
            elif age_minutes < 5:
                age_buckets["1_to_5min"] += 1
            elif age_minutes < 15:
                age_buckets["5_to_15min"] += 1
            elif age_minutes < 60:
                age_buckets["15_to_60min"] += 1
            else:
                age_buckets["over_1hour"] += 1

        return age_buckets


def create_lazy_spec_loader(
    base_spec: Dict[str, Union[str, Dict]],
    llm: BaseLanguageModel,
    cache_dir: Optional[Path] = None,
    **kwargs,
) -> SpecCacheManager:
    """
    Create a lazy spec loader with caching.

    Args:
        base_spec: The complete OpenAPI specification
        llm: The language model to use for intent classification
        cache_dir: Directory for persistent cache storage
        **kwargs: Additional arguments for SpecCacheManager

    Returns:
        Configured SpecCacheManager instance
    """
    if cache_dir is None:
        cache_dir = Path.cwd() / ".spec_cache"
    return SpecCacheManager(base_spec, llm, cache_dir, **kwargs)


def get_spec_with_lazy_loading(
    query: str,
    base_spec: Dict[str, Union[str, Dict]],
    llm: BaseLanguageModel,
    cache_manager: Optional[SpecCacheManager] = None,
) -> Dict[str, Union[str, Dict]]:
    """
    Convenience function to get optimised spec with lazy loading.

    Args:
        query: User query text
        base_spec: Complete OpenAPI specification
        llm: The language model to use for intent classification
        cache_manager: Optional cache manager instance

    Returns:
        Optimised OpenAPI specification
    """
    if cache_manager is None:
        cache_manager = create_lazy_spec_loader(base_spec, llm)

    return cache_manager.get_spec_lazy(query)
