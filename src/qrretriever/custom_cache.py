from typing import Any, Dict, Optional, Tuple
from transformers.cache_utils import DynamicCache, CacheLayerMixin
import torch


class DynamicLayerWithQuery(CacheLayerMixin):
    """
    A single layer's cache that also stores query states for QRRetriever.
    This extends the standard dynamic cache layer to additionally track query states.
    """

    def __init__(self, query_indices=None):
        super().__init__()
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.queries: Optional[torch.Tensor] = None
        self.is_initialized = False
        self.query_indices = query_indices if query_indices is not None else []

    def lazy_initialization(self, key_states: torch.Tensor):
        """Initialize the cache tensors on first use."""
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with new key and value states.

        Parameters:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`Dict[str, Any]`, `optional`): Additional arguments (not used in basic implementation).

        Returns:
            A tuple containing the updated key and value states.
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        # Update key and value caches
        if self.keys is None:
            self.keys = key_states
            self.values = value_states
        else:
            self.keys = torch.cat([self.keys, key_states], dim=-2)
            self.values = torch.cat([self.values, value_states], dim=-2)

        return self.keys, self.values

    def update_with_query(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with new query, key, and value states.

        Parameters:
            query_states (`torch.Tensor`): The new query states to cache.
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`Dict[str, Any]`, `optional`): Additional arguments.

        Returns:
            A tuple containing the updated key and value states.
        """
        # First update keys and values using parent logic
        keys, values = self.update(key_states, value_states, cache_kwargs)

        # Then update query cache
        if query_states is not None:
            if self.queries is None:
                self.queries = query_states
            else:
                self.queries = torch.cat([self.queries, query_states], dim=-2)

        return keys, values

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        if self.keys is None:
            return 0
        return self.keys.shape[-2]

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum cache capacity (None for unlimited)."""
        return None

    def get_mask_sizes(self, cache_position: torch.Tensor) -> Tuple[int, int]:
        """Return the length and offset of the cache, used to generate the mask"""
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_usable_length(self, new_seq_length: int) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        max_length = self.get_max_cache_shape()
        previous_seq_length = self.get_seq_length()
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length


class KeyCacheList:
    """A list-like wrapper that allows direct assignment to key cache layers."""

    def __init__(self, cache):
        self.cache = cache

    def __len__(self):
        return len(self.cache.layers)

    def __getitem__(self, idx):
        return self.cache.layers[idx].keys

    def __setitem__(self, idx, value):
        self.cache.layers[idx].keys = value


class ValueCacheList:
    """A list-like wrapper that allows direct assignment to value cache layers."""

    def __init__(self, cache):
        self.cache = cache

    def __len__(self):
        return len(self.cache.layers)

    def __getitem__(self, idx):
        return self.cache.layers[idx].values

    def __setitem__(self, idx, value):
        self.cache.layers[idx].values = value


class DynamicCacheWithQuery(DynamicCache):
    """
    Cache class used for QRRetriever that also stores query states.
    Extends DynamicCache to track query vectors at specified indices.
    """

    def __init__(self, query_indices=None, config=None, offloading=False, offload_only_non_sliding=False):
        """
        Initialize the cache with query tracking capability.

        Parameters:
            query_indices (list): Indices for query vectors to save.
            config: Model configuration (passed to parent).
            offloading: Whether to offload layers to CPU.
            offload_only_non_sliding: Whether to offload only non-sliding layers.
        """
        self._query_indices = query_indices if query_indices is not None else []

        # Initialize parent DynamicCache without config to avoid layer pre-initialization
        # We'll use lazy initialization with our custom layer class
        super().__init__(config=None, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)

        # Override the layer class to use our custom layer
        self.layer_class_to_replicate = DynamicLayerWithQuery

    @property
    def key_cache(self):
        """Backward compatibility property for accessing keys from all layers."""
        # Return a special list that allows modification
        return KeyCacheList(self)

    @property
    def value_cache(self):
        """Backward compatibility property for accessing values from all layers."""
        # Return a special list that allows modification
        return ValueCacheList(self)

    @property
    def query_cache(self):
        """Backward compatibility property for accessing queries from all layers."""
        return [layer.queries for layer in self.layers]

    @query_cache.setter
    def query_cache(self, value):
        """Setter for clearing or setting query cache."""
        if value == []:
            # Clear query cache for all layers
            for layer in self.layers:
                layer.queries = None

    @property
    def _seen_tokens(self):
        """Backward compatibility property for _seen_tokens (not used in new architecture)."""
        # In the new architecture, we don't track _seen_tokens at the cache level
        # Return the sequence length of the first layer if available
        if len(self.layers) > 0 and self.layers[0].keys is not None:
            return self.layers[0].keys.shape[-2]
        return 0

    @_seen_tokens.setter
    def _seen_tokens(self, value):
        """Setter for _seen_tokens (no-op for backward compatibility)."""
        # In the new architecture, we don't need to track this
        # This setter exists only for backward compatibility
        pass

    def update(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with new query, key, and value states for the layer.

        Parameters:
            query_states (`torch.Tensor`): The new query states to cache.
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            layer_idx (`int`): The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`): Additional arguments for the cache.

        Returns:
            A tuple containing the updated key and value states.
        """
        # Ensure we have enough layers (lazy initialization)
        while len(self.layers) <= layer_idx:
            self.layers.append(DynamicLayerWithQuery(query_indices=self._query_indices))

        # Use the custom update method that handles queries
        layer = self.layers[layer_idx]
        return layer.update_with_query(query_states, key_states, value_states, cache_kwargs)

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Delegate to the layer if it exists
        if layer_idx < len(self.layers):
            return self.layers[layer_idx].get_usable_length(new_seq_length)
        # If layer doesn't exist yet, return 0 (no previous cache)
        return 0

    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        query_indices=None
    ) -> "DynamicCacheWithQuery":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCacheWithQuery`."""
        cache = cls(query_indices=query_indices)
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                # For legacy cache, we don't have query states
                cache.update(None, key_states, value_states, layer_idx)
        return cache
