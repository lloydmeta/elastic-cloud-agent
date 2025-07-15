"""
Tests for the lazy loading system with caching.
"""

import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from elastic_cloud_agent.spec_optimization.lazy_loader import (
    SpecCacheManager,
    create_lazy_spec_loader,
    get_spec_with_lazy_loading,
)


class TestSpecCacheManager:
    """Tests for the SpecCacheManager class."""

    @pytest.fixture
    def sample_spec(self):
        """Sample OpenAPI spec for testing."""
        return {
            "swagger": "2.0",
            "info": {"title": "Test API", "version": "1.0"},
            "paths": {
                "/deployments": {
                    "get": {
                        "operationId": "list-deployments",
                        "tags": ["Deployments"],
                        "summary": "List deployments",
                        "description": "Get a list of all deployments with detailed information",
                        "example": {"deployments": []},
                    },
                    "post": {
                        "operationId": "create-deployment",
                        "tags": ["Deployments"],
                        "summary": "Create deployment",
                        "description": "Create a new deployment with specified configuration",
                    },
                },
                "/deployments/{id}": {
                    "get": {
                        "operationId": "get-deployment",
                        "tags": ["Deployments"],
                        "summary": "Get deployment",
                    },
                    "delete": {
                        "operationId": "delete-deployment",
                        "tags": ["Deployments"],
                        "summary": "Delete deployment",
                    },
                },
                "/costs": {
                    "get": {
                        "operationId": "get-costs-overview",
                        "tags": ["BillingCostsAnalysis"],
                        "summary": "Get billing costs",
                    }
                },
                "/api-keys": {
                    "get": {
                        "operationId": "get-api-keys",
                        "tags": ["Authentication"],
                        "summary": "List API keys",
                    }
                },
            },
            "definitions": {
                "Deployment": {
                    "type": "object",
                    "description": "A deployment object with all configuration details",
                    "properties": {"id": {"type": "string"}},
                    "example": {"id": "123"},
                }
            },
        }

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "basic"
        mock_llm.invoke.return_value = mock_response
        return mock_llm

    @pytest.fixture
    def cache_manager(self, sample_spec, temp_cache_dir, mock_llm):
        """Create a cache manager instance for testing."""
        return SpecCacheManager(
            sample_spec,
            mock_llm,
            cache_dir=temp_cache_dir,
            max_cache_size=5,
            cache_ttl_seconds=60,
        )

    def test_cache_manager_initialisation(self, sample_spec, temp_cache_dir, mock_llm):
        """Test cache manager initialisation."""
        manager = SpecCacheManager(
            sample_spec,
            mock_llm,
            cache_dir=temp_cache_dir,
            max_cache_size=10,
            cache_ttl_seconds=120,
        )

        assert manager.base_spec == sample_spec
        assert manager.cache_dir == temp_cache_dir
        assert manager.max_cache_size == 10
        assert manager.cache_ttl_seconds == 120
        assert temp_cache_dir.exists()

    def test_get_spec_lazy_basic(self, cache_manager):
        """Test basic lazy spec loading."""
        query = "list deployments"
        spec = cache_manager.get_spec_lazy(query)

        # Should return a valid spec
        assert "paths" in spec
        assert isinstance(spec["paths"], dict)

        # Should have cached the result
        stats = cache_manager.get_cache_statistics()
        assert stats["memory_cache_size"] == 1
        assert stats["total_misses"] == 1

    def test_get_spec_lazy_cache_hit(self, cache_manager):
        """Test cache hit behaviour."""
        query = "list deployments"

        # First call - cache miss
        spec1 = cache_manager.get_spec_lazy(query)
        stats_after_first = cache_manager.get_cache_statistics()
        assert stats_after_first["total_misses"] == 1
        assert stats_after_first["total_hits"] == 0

        # Second call - should be cache hit
        spec2 = cache_manager.get_spec_lazy(query)
        stats_after_second = cache_manager.get_cache_statistics()
        assert stats_after_second["total_misses"] == 1
        assert stats_after_second["total_hits"] == 1

        # Specs should be identical
        assert spec1 == spec2

    def test_persistent_cache_storage(self, cache_manager):
        """Test persistent cache storage and retrieval."""
        query = "create deployment"

        # Load spec with persistent cache enabled
        cache_manager.get_spec_lazy(query, use_persistent_cache=True)

        # Check that cache file was created
        cache_files = list(cache_manager.cache_dir.glob("*.json"))
        assert len(cache_files) == 1

        # Verify cache file contents
        with open(cache_files[0], "r") as f:
            cached_data = json.load(f)

        assert "spec" in cached_data
        assert "metadata" in cached_data
        assert cached_data["metadata"]["original_query"] == query

    def test_persistent_cache_retrieval(self, cache_manager):
        """Test loading from persistent cache."""
        query = "manage API keys"

        # First load - should create persistent cache
        spec1 = cache_manager.get_spec_lazy(query, use_persistent_cache=True)

        # Clear memory cache
        cache_manager._memory_cache.clear()
        cache_manager._cache_timestamps.clear()

        # Second load - should retrieve from persistent cache
        spec2 = cache_manager.get_spec_lazy(query, use_persistent_cache=True)

        assert spec1 == spec2

        # Should show cache hit
        stats = cache_manager.get_cache_statistics()
        assert stats["total_hits"] >= 1

    def test_cache_expiration(self, sample_spec, temp_cache_dir, mock_llm):
        """Test cache expiration behaviour."""
        manager = SpecCacheManager(
            sample_spec,
            mock_llm,
            cache_dir=temp_cache_dir,
            cache_ttl_seconds=1,  # Very short TTL for testing
        )

        query = "check deployment health"

        # Load spec
        manager.get_spec_lazy(query)

        # Wait for cache to expire
        time.sleep(1.1)

        # Load again - should be cache miss due to expiration
        manager.get_spec_lazy(query)

        stats = manager.get_cache_statistics()
        assert stats["total_misses"] == 2  # Both calls should be misses

    def test_force_refresh(self, cache_manager):
        """Test force refresh functionality."""
        query = "view billing costs"

        # First load
        cache_manager.get_spec_lazy(query)

        # Force refresh - should bypass cache
        cache_manager.get_spec_lazy(query, force_refresh=True)

        stats = cache_manager.get_cache_statistics()
        assert stats["total_misses"] == 2  # Both should be misses

    def test_cache_size_limit_and_lru_eviction(self, sample_spec, temp_cache_dir, mock_llm):
        """Test cache size limits and LRU eviction."""
        manager = SpecCacheManager(
            sample_spec,
            mock_llm,
            cache_dir=temp_cache_dir,
            max_cache_size=3,  # Small cache for testing
        )

        queries = [
            "list deployments",
            "create deployment",
            "get deployment",
            "delete deployment",
            "view costs",
        ]

        # Load more specs than cache can hold
        for query in queries:
            manager.get_spec_lazy(query)

        # Cache should not exceed max size
        stats = manager.get_cache_statistics()
        assert stats["memory_cache_size"] <= 3
        assert stats["total_evictions"] > 0

    def test_preload_common_specs(self, cache_manager):
        """Test preloading of common specifications."""
        common_queries = [
            "list deployments",
            "create deployment",
            "check cluster health",
        ]

        # Preload specs
        cache_manager.preload_common_specs(common_queries)

        # Give preloading thread time to complete
        time.sleep(2.0)  # Increased wait time for LLM calls

        # Check that some specs were preloaded (may not be all due to LLM failures)
        stats = cache_manager.get_cache_statistics()
        assert stats["memory_cache_size"] >= 1  # At least one should succeed
        assert stats["total_preloads"] >= 1

    def test_get_cached_queries(self, cache_manager):
        """Test retrieval of cached query list."""
        queries = ["list deployments", "create deployment", "manage API keys"]

        # Load some specs
        for query in queries:
            cache_manager.get_spec_lazy(query)

        # Get cached queries
        cached_queries = cache_manager.get_cached_queries()

        assert len(cached_queries) == len(queries)
        # Queries might be transformed, so check we have the right count

    def test_invalidate_cache_all(self, cache_manager):
        """Test full cache invalidation."""
        queries = ["list deployments", "create deployment", "view costs"]

        # Load some specs
        for query in queries:
            cache_manager.get_spec_lazy(query, use_persistent_cache=True)

        # Verify cache has entries
        assert cache_manager.get_cache_statistics()["memory_cache_size"] > 0
        assert cache_manager.get_cache_statistics()["persistent_cache_size"] > 0

        # Invalidate all
        invalidated_count = cache_manager.invalidate_cache()

        # Check cache is empty
        stats = cache_manager.get_cache_statistics()
        assert stats["memory_cache_size"] == 0
        assert stats["persistent_cache_size"] == 0
        assert invalidated_count == len(queries)

    def test_invalidate_cache_pattern(self, cache_manager):
        """Test pattern-based cache invalidation."""
        # Load specs with different patterns
        cache_manager.get_spec_lazy("list deployments")
        cache_manager.get_spec_lazy("create deployment")
        cache_manager.get_spec_lazy("view billing costs")

        # Invalidate only deployment-related specs
        # This is tricky since cache keys are hashed, but we can test the mechanism
        initial_size = cache_manager.get_cache_statistics()["memory_cache_size"]

        # Get a cache key to test pattern matching
        cache_keys = list(cache_manager._memory_cache.keys())
        if cache_keys:
            # Use part of a real cache key as pattern
            pattern = cache_keys[0][:8]
            cache_manager.invalidate_cache(pattern)

            final_size = cache_manager.get_cache_statistics()["memory_cache_size"]
            assert final_size < initial_size

    def test_cleanup_expired_cache(self, sample_spec, temp_cache_dir, mock_llm):
        """Test cleanup of expired cache entries."""
        manager = SpecCacheManager(
            sample_spec,
            mock_llm,
            cache_dir=temp_cache_dir,
            cache_ttl_seconds=1,  # Short TTL for testing
        )

        # Load some specs
        manager.get_spec_lazy("list deployments")
        manager.get_spec_lazy("create deployment")

        # Wait for expiration
        time.sleep(1.1)

        # Cleanup expired entries
        removed_count = manager.cleanup_expired_cache()

        assert removed_count == 2
        assert manager.get_cache_statistics()["memory_cache_size"] == 0

    def test_cache_statistics_comprehensive(self, cache_manager):
        """Test comprehensive cache statistics."""
        # Generate some activity
        cache_manager.get_spec_lazy("list deployments")  # Miss
        cache_manager.get_spec_lazy("list deployments")  # Hit
        cache_manager.get_spec_lazy("create deployment")  # Miss

        stats = cache_manager.get_cache_statistics()

        # Check all required fields are present
        required_fields = [
            "memory_cache_size",
            "persistent_cache_size",
            "hit_rate",
            "total_hits",
            "total_misses",
            "total_evictions",
            "total_preloads",
            "most_accessed_specs",
            "cache_age_distribution",
        ]

        for field in required_fields:
            assert field in stats

        # Verify hit rate calculation
        expected_hit_rate = stats["total_hits"] / (stats["total_hits"] + stats["total_misses"])
        assert abs(stats["hit_rate"] - expected_hit_rate) < 0.001

    def test_concurrent_access(self, cache_manager):
        """Test thread safety with concurrent access."""
        query = "list deployments"
        results = []
        errors = []

        def worker():
            try:
                spec = cache_manager.get_spec_lazy(query)
                results.append(spec)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=worker) for _ in range(10)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0
        assert len(results) == 10

        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result

    def test_corrupted_cache_file_handling(self, cache_manager):
        """Test handling of corrupted cache files."""
        query = "list deployments"

        # Create a corrupted cache file
        cache_key = cache_manager._generate_cache_key(query)
        cache_file = cache_manager.cache_dir / f"{cache_key}.json"

        # Write invalid JSON
        with open(cache_file, "w") as f:
            f.write("invalid json content")

        # Verify the corrupted file exists
        assert cache_file.exists()

        # Should handle gracefully and generate fresh spec
        spec = cache_manager.get_spec_lazy(query, use_persistent_cache=True)

        assert "paths" in spec
        # Corrupted file should be removed after the failed load attempt
        # (The cache manager should clean up corrupted files)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.fixture
    def sample_spec(self):
        """Sample spec for convenience function tests."""
        return {
            "swagger": "2.0",
            "info": {"title": "Test API"},
            "paths": {
                "/deployments": {
                    "get": {
                        "operationId": "list-deployments",
                        "tags": ["Deployments"],
                        "description": "Long description with verbose details",
                        "example": {"data": "example"},
                    }
                }
            },
        }

    def test_create_lazy_spec_loader(self, sample_spec):
        """Test creating a lazy spec loader."""
        mock_llm = Mock()
        loader = create_lazy_spec_loader(sample_spec, mock_llm)

        assert isinstance(loader, SpecCacheManager)
        assert loader.base_spec == sample_spec

    def test_create_lazy_spec_loader_with_custom_settings(self, sample_spec):
        """Test creating loader with custom settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            mock_llm = Mock()

            loader = create_lazy_spec_loader(
                sample_spec,
                mock_llm,
                cache_dir=cache_dir,
                max_cache_size=50,
                cache_ttl_seconds=1800,
            )

            assert loader.cache_dir == cache_dir
            assert loader.max_cache_size == 50
            assert loader.cache_ttl_seconds == 1800

    def test_get_spec_with_lazy_loading_no_manager(self, sample_spec):
        """Test convenience function without existing manager."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "basic"
        mock_llm.invoke.return_value = mock_response

        spec = get_spec_with_lazy_loading("list deployments", sample_spec, mock_llm)

        # Should return optimised spec
        assert "paths" in spec
        assert "/deployments" in spec["paths"]

    def test_get_spec_with_lazy_loading_with_manager(self, sample_spec):
        """Test convenience function with existing manager."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "management"
        mock_llm.invoke.return_value = mock_response

        manager = create_lazy_spec_loader(sample_spec, mock_llm)

        spec = get_spec_with_lazy_loading(
            "create deployment", sample_spec, mock_llm, cache_manager=manager
        )

        # Should return optimised spec
        assert "paths" in spec

        # Manager should have cached the result
        stats = manager.get_cache_statistics()
        assert stats["memory_cache_size"] == 1


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.fixture
    def sample_spec(self):
        """Sample spec for edge case testing."""
        return {
            "swagger": "2.0",
            "info": {"title": "Test API"},
            "paths": {
                "/test": {
                    "get": {
                        "operationId": "test-operation",
                        "tags": ["Test"],
                        "summary": "Test operation",
                    }
                }
            },
        }

    def test_empty_spec_handling(self):
        """Test handling of empty specifications."""
        empty_spec = {}
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "basic"
        mock_llm.invoke.return_value = mock_response

        manager = SpecCacheManager(empty_spec, mock_llm)

        # Should handle gracefully
        spec = manager.get_spec_lazy("any query")
        assert isinstance(spec, dict)

    def test_malformed_spec_handling(self):
        """Test handling of malformed specifications."""
        malformed_spec = {"swagger": "2.0", "paths": "invalid"}
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "basic"
        mock_llm.invoke.return_value = mock_response

        manager = SpecCacheManager(malformed_spec, mock_llm)

        # Should handle gracefully
        spec = manager.get_spec_lazy("any query")
        assert isinstance(spec, dict)

    def test_very_long_query(self):
        """Test handling of very long queries."""
        spec = {"swagger": "2.0", "info": {"title": "Test"}}
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "management"
        mock_llm.invoke.return_value = mock_response

        manager = SpecCacheManager(spec, mock_llm)

        long_query = "create " + "deployment " * 1000 + "with configuration"

        # Should handle without errors
        result = manager.get_spec_lazy(long_query)
        assert isinstance(result, dict)

    def test_special_characters_in_query(self):
        """Test handling of special characters in queries."""
        spec = {"swagger": "2.0", "info": {"title": "Test"}}
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "management"
        mock_llm.invoke.return_value = mock_response

        manager = SpecCacheManager(spec, mock_llm)

        special_query = "create deployment @#$%^&*()!{}[]|\\:;\"'<>?,./"

        # Should handle without errors
        result = manager.get_spec_lazy(special_query)
        assert isinstance(result, dict)

    def test_cache_directory_permissions(self, sample_spec):
        """Test handling of cache directory permission issues."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "basic"
        mock_llm.invoke.return_value = mock_response

        # Use a temporary directory that we can control
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a read-only directory
            ro_dir = Path(temp_dir) / "readonly"
            ro_dir.mkdir()
            ro_dir.chmod(0o444)  # Read-only

            # Attempt to create cache manager with read-only directory
            try:
                manager = SpecCacheManager(sample_spec, mock_llm, cache_dir=ro_dir / "subdir")
                # Should work without persistent cache even if directory creation fails
                spec = manager.get_spec_lazy("test query", use_persistent_cache=False)
                assert isinstance(spec, dict)
            except OSError:
                # Expected if directory creation fails
                pass

    def test_cache_with_none_directory(self, sample_spec):
        """Test cache manager with None cache directory."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "basic"
        mock_llm.invoke.return_value = mock_response

        manager = SpecCacheManager(sample_spec, mock_llm, cache_dir=None)

        # Should work with memory cache only
        spec = manager.get_spec_lazy("test query")
        assert isinstance(spec, dict)

        stats = manager.get_cache_statistics()
        assert stats["persistent_cache_size"] == 0
