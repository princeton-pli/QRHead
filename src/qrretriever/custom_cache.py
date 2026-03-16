from typing import Any, Dict, List, Optional, Tuple
from transformers.cache_utils import DynamicCache, DynamicLayer
import torch


class DynamicLayerWithQuery(DynamicLayer):
    """DynamicLayer extended to also accumulate query states."""

    def __init__(self) -> None:
        super().__init__()
        self.queries: Optional[torch.Tensor] = None

    def update_query(self, query_states: torch.Tensor) -> None:
        if self.queries is None or self.queries.numel() == 0:
            self.queries = query_states
        else:
            self.queries = torch.cat([self.queries, query_states], dim=-2)


class DynamicCacheWithQuery(DynamicCache):
    """
    Cache class used for QRRetriever — updated for transformers 5.x.
    
    5.x replaced the flat key_cache/value_cache lists with a self.layers[]
    list of DynamicLayer objects. This class:
      - keeps the identical update() signature from 4.44.1 (query, key, value, layer_idx)
      - re-exposes key_cache / value_cache as properties for any code that
        accesses them directly (e.g. model attention implementations)
      - stores query states inside DynamicLayerWithQuery per layer
    """

    def __init__(self, query_indices: List[int] = []) -> None:
        super().__init__()
        self._query_indices = query_indices

    # ------------------------------------------------------------------
    # Compatibility properties — these are what 4.44.1 exposed as plain
    # list attributes. Now they read through to self.layers[].
    # ------------------------------------------------------------------
    @property
    def key_cache(self) -> List[torch.Tensor]:
        return [
            layer.keys if (layer.is_initialized and layer.keys.numel() > 0)
            else []
            for layer in self.layers
        ]

    @property
    def value_cache(self) -> List[torch.Tensor]:
        return [
            layer.values if (layer.is_initialized and layer.values.numel() > 0)
            else []
            for layer in self.layers
        ]

    # ------------------------------------------------------------------
    # query_cache — mirrors the old list interface
    # ------------------------------------------------------------------
    @property
    def query_cache(self) -> List[Optional[torch.Tensor]]:
        return [getattr(layer, "queries", None) for layer in self.layers]

    # ------------------------------------------------------------------
    # update() — identical signature to 4.44.1, no call-site changes needed
    # ------------------------------------------------------------------
    def update(
        self,
        query_states: Optional[torch.Tensor],
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ensure layers list is long enough, using our extended layer type
        while len(self.layers) <= layer_idx:
            self.layers.append(DynamicLayerWithQuery())

        layer: DynamicLayerWithQuery = self.layers[layer_idx]

        # key/value update via the layer object (handles lazy init + concat)
        keys, values = layer.update(key_states, value_states, cache_kwargs)

        # query update
        if query_states is not None:
            layer.update_query(query_states)

        return keys, values

    # ------------------------------------------------------------------
    # from_legacy_cache — removed in 5.x, reimplemented here
    # ------------------------------------------------------------------
    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        query_indices: List[int] = [],
    ) -> "DynamicCacheWithQuery":
        cache = cls(query_indices=query_indices)
        if past_key_values is not None:
            for layer_idx, (key_states, value_states) in enumerate(past_key_values):
                cache.update(None, key_states, value_states, layer_idx)
        return cache
