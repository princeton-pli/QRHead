# To add models (use Transformers 5.x)

-  cp transformers_source code/src/models/XXX/modeling_XXX.py custom_modeling_XXX.py
- Fix the relative import to actual transformer imports
- import DynamicCacheWithQuery from .custom_cache
- search from past_key_values.updates and add these lines
    if isinstance(past_key_values, DynamicCacheWithQuery):
        query_states_to_cache = query_states[:, :, past_key_values._query_indices, :]
        key_states, value_states = past_key_values.update(query_states_to_cache, key_states, value_states, self.layer_idx, cache_kwargs)
    else:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
- search for logits = ... and 
    if compute_logits:
        logits = ....
    else:
        logits = None
    
    and compute_logits: Optional[bool]=True in the functions defintion. 


# in attn_retriever add the necesaary things
    # BASE_CLASS
    # prompt_prefix
    # prompt_separator accordingly