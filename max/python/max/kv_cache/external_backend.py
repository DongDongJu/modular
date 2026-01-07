# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""External KV Cache Backend Interface.

This module defines the interface for integrating external KV cache systems
(like LMCache) with MAX's paged KV cache manager.

Example:
    >>> from max.kv_cache.external_backend import create_external_backend
    >>> 
    >>> config = {
    ...     "type": "lmcache",
    ...     "chunk_size": 256,
    ...     "enable_cpu_cache": True,
    ... }
    >>> backend = create_external_backend(config)
    >>> 
    >>> # Lookup cached KV
    >>> result = backend.lookup(tokens)
    >>> if result.matched_prefix_len > 0:
    ...     backend.load(tokens, result.matched_prefix_len, blocks, tensors)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:
    from max.driver import Tensor

logger = logging.getLogger("max.kv_cache.external")


@dataclass
class ExternalCacheLookupResult:
    """Result of an external cache lookup.
    
    Attributes:
        matched_prefix_len: Number of tokens with cached KV available.
            This value is aligned to page boundaries.
        cache_tier: Which tier the cache was found in (e.g., "gpu", "cpu", "disk", "remote").
        metadata: Backend-specific metadata for the load operation.
    """
    matched_prefix_len: int
    cache_tier: Optional[str] = None
    metadata: Optional[Any] = None


@dataclass
class ExternalCacheLoadResult:
    """Result of loading KV from external cache.
    
    Attributes:
        success: Whether the load operation succeeded.
        loaded_tokens: Number of tokens actually loaded.
        load_time_ms: Time taken to load in milliseconds.
        error: Error message if the load failed.
    """
    success: bool
    loaded_tokens: int
    load_time_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class ExternalCacheStoreResult:
    """Result of storing KV to external cache.
    
    Attributes:
        event_id: Async event ID for tracking completion.
        stored_tokens: Number of tokens queued for storage.
    """
    event_id: str
    stored_tokens: int


@dataclass
class ExternalCacheStats:
    """Statistics from external cache backend.
    
    Attributes:
        hits: Number of cache hits.
        misses: Number of cache misses.
        hit_rate: Cache hit rate (0.0 to 1.0).
        bytes_loaded: Total bytes loaded from cache.
        bytes_stored: Total bytes stored to cache.
        tier_hits: Hits per cache tier.
    """
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    bytes_loaded: int = 0
    bytes_stored: int = 0
    tier_hits: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "bytes_loaded": self.bytes_loaded,
            "bytes_stored": self.bytes_stored,
            "tier_hits": self.tier_hits,
        }


class ExternalKVCacheBackend(ABC):
    """Interface for external KV cache backends.
    
    Implementations of this interface allow MAX to use external storage
    systems (like LMCache) for KV cache persistence and sharing.
    
    The interface is designed to be:
    - Non-blocking where possible (async store)
    - Page-aligned (respects MAX's page_size)
    - Scheduler-thread safe (called from MAX's scheduler)
    
    Subclasses must implement:
    - lookup(): Check if KV cache exists for tokens
    - load(): Load KV data from external cache
    - store(): Store KV data to external cache (async)
    - is_store_complete(): Check if async store is done
    
    Example:
        >>> class MyBackend(ExternalKVCacheBackend):
        ...     def lookup(self, tokens):
        ...         # Check external cache
        ...         return ExternalCacheLookupResult(matched_prefix_len=128)
        ...     
        ...     def load(self, tokens, prefix_len, dst_blocks, dst_tensors):
        ...         # Load KV from external cache
        ...         return ExternalCacheLoadResult(success=True, loaded_tokens=128)
        ...     
        ...     # ... implement other methods
    """
    
    @abstractmethod
    def lookup(self, tokens: Sequence[int]) -> ExternalCacheLookupResult:
        """Check if KV cache exists for the given token sequence.
        
        This method queries the external cache to determine how many
        prefix tokens have cached KV data available.
        
        Args:
            tokens: Input token IDs.
            
        Returns:
            Lookup result with matched prefix length (must be page-aligned).
            
        Note:
            The returned matched_prefix_len should be aligned to MAX's
            page_size boundary. For example, if 200 tokens match but
            page_size is 128, return 128 (floor to page boundary).
        """
        ...
    
    @abstractmethod
    def load(
        self,
        tokens: Sequence[int],
        prefix_len: int,
        dst_blocks: Sequence[int],
        dst_tensors: Sequence[Tensor],
    ) -> ExternalCacheLoadResult:
        """Load KV data from external cache into MAX's paged buffers.
        
        This method copies KV cache data from the external storage
        into the specified blocks of MAX's KV cache tensors.
        
        Args:
            tokens: Full token sequence (for cache key lookup).
            prefix_len: Number of prefix tokens to load (page-aligned).
            dst_blocks: Block indices in MAX's cache to load into.
            dst_tensors: KV cache tensors, one per device in tensor parallelism.
                Each tensor has shape [total_pages, kv_dim, num_layers, page_size, n_heads, head_dim].
            
        Returns:
            Load result indicating success/failure and tokens loaded.
            
        Note:
            This method should be synchronous - it blocks until the load
            is complete, as MAX needs the data immediately for prefill.
        """
        ...
    
    @abstractmethod
    def store(
        self,
        tokens: Sequence[int],
        src_blocks: Sequence[int],
        src_tensors: Sequence[Tensor],
        start_pos: int = 0,
    ) -> ExternalCacheStoreResult:
        """Store KV data to external cache (async).
        
        This method queues KV cache data to be stored in the external
        storage system. The store operation should be asynchronous
        to avoid blocking the inference pipeline.
        
        Args:
            tokens: Token sequence being stored.
            src_blocks: Block indices in MAX's cache to read from.
            src_tensors: KV cache tensors, one per device in tensor parallelism.
            start_pos: Starting token position (for incremental store,
                e.g., after loading prefix from cache).
            
        Returns:
            Store result with async event ID for tracking completion.
            
        Note:
            Use is_store_complete() to check if the store has finished.
            Blocks should not be freed until the store is complete.
        """
        ...
    
    @abstractmethod
    def is_store_complete(self, event_id: str) -> bool:
        """Check if an async store operation has completed.
        
        Args:
            event_id: Event ID returned from a previous store() call.
            
        Returns:
            True if the store operation has completed, False otherwise.
        """
        ...
    
    def get_stats(self) -> ExternalCacheStats:
        """Get backend statistics.
        
        Returns:
            Statistics including hit rate, bytes transferred, etc.
        """
        return ExternalCacheStats()
    
    def reset_stats(self) -> None:
        """Reset backend statistics."""
        pass
    
    def shutdown(self) -> None:
        """Clean up backend resources.
        
        Called when the KV cache manager is being destroyed.
        Implementations should wait for pending stores to complete
        and release any held resources.
        """
        pass


class NullExternalKVCacheBackend(ExternalKVCacheBackend):
    """No-op backend for testing and when external cache is disabled.
    
    All lookups return cache miss (matched_prefix_len=0).
    All loads and stores are no-ops that succeed immediately.
    
    This backend is used:
    - When no external backend is configured
    - For unit testing without LMCache dependency
    - As a fallback when the configured backend fails to initialize
    """
    
    def __init__(self) -> None:
        """Initialize the null backend."""
        self._lookup_count = 0
        self._load_count = 0
        self._store_count = 0
    
    def lookup(self, tokens: Sequence[int]) -> ExternalCacheLookupResult:
        """Always returns cache miss.
        
        Args:
            tokens: Input token IDs (ignored).
            
        Returns:
            Lookup result with matched_prefix_len=0.
        """
        self._lookup_count += 1
        return ExternalCacheLookupResult(matched_prefix_len=0)
    
    def load(
        self,
        tokens: Sequence[int],
        prefix_len: int,
        dst_blocks: Sequence[int],
        dst_tensors: Sequence[Tensor],
    ) -> ExternalCacheLoadResult:
        """No-op load that always succeeds.
        
        Args:
            tokens: Token sequence (ignored).
            prefix_len: Prefix length (ignored).
            dst_blocks: Block indices (ignored).
            dst_tensors: KV tensors (ignored).
            
        Returns:
            Success result with loaded_tokens=0.
        """
        self._load_count += 1
        return ExternalCacheLoadResult(success=True, loaded_tokens=0)
    
    def store(
        self,
        tokens: Sequence[int],
        src_blocks: Sequence[int],
        src_tensors: Sequence[Tensor],
        start_pos: int = 0,
    ) -> ExternalCacheStoreResult:
        """No-op store that always succeeds.
        
        Args:
            tokens: Token sequence (ignored).
            src_blocks: Block indices (ignored).
            src_tensors: KV tensors (ignored).
            start_pos: Start position (ignored).
            
        Returns:
            Result with event_id="null" and stored_tokens=0.
        """
        self._store_count += 1
        return ExternalCacheStoreResult(event_id="null", stored_tokens=0)
    
    def is_store_complete(self, event_id: str) -> bool:
        """Always returns True (stores complete immediately).
        
        Args:
            event_id: Event ID (ignored).
            
        Returns:
            True.
        """
        return True
    
    def get_stats(self) -> ExternalCacheStats:
        """Get null backend statistics.
        
        Returns:
            Stats with all zeros except for operation counts.
        """
        return ExternalCacheStats(
            hits=0,
            misses=self._lookup_count,
            hit_rate=0.0,
        )


# Backend registry for dynamic creation
_BACKEND_REGISTRY: dict[str, type[ExternalKVCacheBackend]] = {
    "null": NullExternalKVCacheBackend,
}


def register_external_backend(
    name: str, 
    backend_class: type[ExternalKVCacheBackend]
) -> None:
    """Register an external backend type.
    
    Args:
        name: Name to register the backend under.
        backend_class: Backend class to register.
        
    Example:
        >>> from max.kv_cache.external_backend import register_external_backend
        >>> register_external_backend("my_backend", MyBackendClass)
    """
    _BACKEND_REGISTRY[name] = backend_class


def create_external_backend(
    config: Optional[dict] = None,
) -> ExternalKVCacheBackend:
    """Create an external KV cache backend from configuration.
    
    Args:
        config: Configuration dictionary with:
            - "type": Backend type name (default: "null")
            - Other backend-specific options
            
    Returns:
        Configured backend instance.
        
    Example:
        >>> config = {
        ...     "type": "lmcache",
        ...     "chunk_size": 256,
        ...     "enable_cpu_cache": True,
        ... }
        >>> backend = create_external_backend(config)
    """
    if config is None:
        return NullExternalKVCacheBackend()
    
    backend_type = config.get("type", "null")
    
    # Check registry first
    if backend_type in _BACKEND_REGISTRY:
        backend_class = _BACKEND_REGISTRY[backend_type]
        return backend_class(**{k: v for k, v in config.items() if k != "type"})
    
    # Try dynamic import for known types
    if backend_type == "lmcache":
        try:
            from lmcache.integration.max.max_external_backend import (
                LMCacheExternalBackend,
            )
            # Pass config to LMCacheExternalBackend
            backend_config = {k: v for k, v in config.items() if k != "type"}
            return LMCacheExternalBackend(**backend_config)
        except ImportError:
            logger.warning(
                "LMCache not installed. Falling back to null backend. "
                "Install with: pip install lmcache"
            )
            return NullExternalKVCacheBackend()
    
    logger.warning(f"Unknown backend type: {backend_type}. Using null backend.")
    return NullExternalKVCacheBackend()
