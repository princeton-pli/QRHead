import gc
from typing import Dict, List, Optional, Tuple, Union
import transformers
import torch
import math
from pathlib import Path

from .config import load_config
from .custom_cache import DynamicCacheWithQuery
from .custom_modeling_llama import LlamaForCausalLM, repeat_kv
from .custom_modeling_qwen2 import Qwen2ForCausalLM
from .custom_modeling_smollmv3 import SmolLM3ForCausalLM

PACKAGE_DIR = Path(__file__).parent
CONFIG_DIR = PACKAGE_DIR / 'configs'

class SPEC_HEAD_SET:
    """
    A set of fixed attention heads.
    """
    FULL_SET = "full_heads"

# attention based retriever
class AttnBasedRetriever:
    # core needs: model_name_or_path, model_base_class, attention_head_subset
    def __init__(self,
        config_or_config_path: Optional[Union[Dict, str]] = None,
        model_name_or_path: Optional[str] = None,
        model_base_class: Optional[str] = None,
        attn_head_set: Optional[str] = None, # 1-2,3-4,5-6
        device: Optional[str] = None,
        # controling retriever
        null_query: Optional[str] = "N/A",
    ):
        config = self.setup_config(config_or_config_path, model_name_or_path, model_base_class)
        print("Using config:", config)
        self.config = config
        # override config with kwargs
        for k, v in config.items():
            setattr(self, k, v)

        # init model etc
        if self.model_base_class.lower() in ['llama-3.1-8b-instruct', 'llama-3.1-70b-instruct', 'llama-3.2-3b-instruct', 'llama-3.2-1b-instruct']:
            BaseClass = LlamaForCausalLM
        elif self.model_base_class.lower() in ['qwen2.5-7b-instruct']:
            BaseClass = Qwen2ForCausalLM
        elif self.model_base_class.lower() in ["smollm3forcausallm"]:
            BaseClass = SmolLM3ForCausalLM
        else:
            raise ValueError(f"Unsupported model class: {self.model_base_class}")
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.llm = BaseClass.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16, 
            attn_implementation="flash_attention_2",
            device_map='auto'
        )
        self.llm.config.pad_token_id = self.llm.config.eos_token_id

        self.start_layer = 0
        self.end_layer = self.llm.config.num_hidden_layers - 1

        # override config if attn_head_set is provided
        if attn_head_set is not None:
            self.attn_head_set = attn_head_set

        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.null_query = null_query

    def setup_config(self,
        config_or_config_path: Optional[Union[Dict, str]] = None,
        model_name_or_path: Optional[str] = None,
        model_base_class: Optional[str] = None,
    ):
        if config_or_config_path is not None:
            assert isinstance(config_or_config_path, (str, Dict))
            if isinstance(config_or_config_path, str):
                config = load_config(config_or_config_path)
            else:
                config = config_or_config_path
        else:
            # infer from model information
            if model_base_class is not None:
                # TODO: infer config from model_base_class
                raise NotImplementedError()
            elif model_name_or_path is not None:
                # TODO: infer config from model_name_or_path
                raise NotImplementedError()
            else:
                raise ValueError("model_name_or_path or model_base_class is required to use default config")
        return config

    def get_content_span(self, prompt: str, char_offset_to_token_idx: dict, content: str):
        if content not in prompt:
            raise ValueError(f"Content not found in the prompt.")
        
        content_char_start = prompt.index(content)
        content_char_end = content_char_start + len(content) - 1 # inclusive end index

        # find the token indices that correspond to the character indices
        start_idx = char_offset_to_token_idx[content_char_start]
        end_idx = char_offset_to_token_idx[content_char_end]
        return start_idx, end_idx
    
    def get_prompt(self, query: str, docs: List[Dict]):
        if self.model_base_class.lower() in ['llama-3.1-8b-instruct', 'llama-3.1-70b-instruct', 'llama-3.2-3b-instruct', 'llama-3.2-1b-instruct']:
            self.prompt_prefix = '<|start_header_id|>user<|end_header_id|>'
            self.prompt_suffix = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        elif self.model_base_class.lower() in ['qwen2.5-7b-instruct']:
            self.prompt_prefix = '<|im_start|>user'
            self.prompt_suffix = '<|im_end|>\n<|im_start|>assistant'
        elif self.model_base_class.lower() in ["smollm3forcausallm"]:
            self.prompt_prefix = '<|im_start|>system\n## Metadata\n\nKnowledge Cutoff Date: June 2025\nToday Date: 05 March 2026\nReasoning Mode: /think\n\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracking, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion.\n\n<|im_start|>user'
            self.prompt_suffix = '<|im_end|>\n<|im_start|>assistant'
        else:
            raise NotImplementedError("Prompt prefix and suffix not defined for the model of {}.".format(self.model_base_class))
        
        if self.model_base_class.lower() in ['llama-3.1-8b-instruct', 'llama-3.1-70b-instruct', 'llama-3.2-3b-instruct', 'llama-3.2-1b-instruct']:
            self.prompt_separator = ' \n\n'
        elif self.model_base_class.lower() in ['qwen2.5-7b-instruct']:
            self.prompt_separator = '\n\n'
        elif self.model_base_class.lower() in ["smollm3forcausallm"]:
            self.prompt_separator = '\n'
        else:
            self.prompt_separator = '\n\n'

        self.retrieval_instruction = ' Here are some paragraphs:'
        self.retrieval_instruction_late = 'Please find information that are relevant to the following query in the paragraphs above.'
        
        llm_prompt = self.prompt_prefix + self.retrieval_instruction
        for i, doc in enumerate(docs):

            paragraph_text = doc['paragraph_text']
            if doc.get('title', None) is not None:
                paragraph_text = doc['title'] + '\n' + paragraph_text

            doc = f'[{i+1}] {paragraph_text}'
            llm_prompt += self.prompt_separator + doc

        llm_prompt += self.prompt_separator + self.retrieval_instruction_late + self.prompt_separator + 'Query:'
        query_prompt = f' {query}' + self.prompt_suffix
        llm_prompt += query_prompt
        return llm_prompt
    
    def compose_scoring_prompt(self, query: str, docs: List[Dict]) -> Tuple:
        llm_prompt = self.get_prompt(query, docs)

        prompt_tokenization_output = self.tokenizer(llm_prompt, return_offsets_mapping=True)
        prompt_token_ids = prompt_tokenization_output["input_ids"]
        offset_mapping = prompt_tokenization_output["offset_mapping"]

        # create mapping from character offset to token index
        char_offset_to_token_idx = {}
        for i, (start, end) in enumerate(offset_mapping):
            for j in range(start, end):
                char_offset_to_token_idx[j] = i

        document_span_intervals = []
        for i, doc in enumerate(docs):
            paragraph_text = doc['paragraph_text']
            if doc.get('title', None) is not None:
                paragraph_text = doc['title'] + '\n' + paragraph_text

            doc_content = f'[{i+1}] {paragraph_text}'
            start_idx, end_idx = self.get_content_span(llm_prompt, char_offset_to_token_idx, doc_content)
            document_span_intervals.append((start_idx, end_idx))

        query_content = self.retrieval_instruction_late + self.prompt_separator + 'Query:' + f' {query}' + self.prompt_suffix
        query_start_idx, query_end_idx = self.get_content_span(llm_prompt, char_offset_to_token_idx, query_content)
        query_span = (query_start_idx, query_end_idx)

        return llm_prompt, prompt_token_ids, query_span, document_span_intervals            
    
    def _trim_kv_cache_to(self, kv_cache: DynamicCacheWithQuery, seq_len: int) -> None:
        """
        Trim all cached key/value tensors to `seq_len` tokens in-place,
        and reset the internal seen-token counter.

        In transformers 5.x DynamicCache stores per-layer DynamicLayer objects
        in self.layers[]; there are no key_cache / value_cache list attributes.
        """
        for layer in kv_cache.layers:
            if layer.is_initialized:
                layer.keys = layer.keys[:, :, :seq_len, :]
                layer.values = layer.values[:, :, :seq_len, :]
        # _seen_tokens is still present on the Cache base class in 5.x
        kv_cache._seen_tokens = seq_len

    def _reset_query_cache(self, kv_cache: DynamicCacheWithQuery, query_indices: List[int]) -> None:
        """
        Clear accumulated query states from all layers and update the
        query-index list for the next forward pass.
        """
        for layer in kv_cache.layers:
            if hasattr(layer, 'queries'):
                layer.queries = None
        kv_cache._query_indices = query_indices

    def score_docs(self, query: str, docs: List[Dict]) -> Dict[str, Dict]:
        prompt, tokenized_prompt, query_span, doc_spans = self.compose_scoring_prompt(query, docs)
        null_prompt, tokenized_null_prompt, null_query_span, _ = self.compose_scoring_prompt(self.null_query, docs)

        assert tokenized_null_prompt[:query_span[0]] == tokenized_prompt[:query_span[0]]
        assert query_span[0] == null_query_span[0], "Query start indices do not match between query and null query."

        # scoring with actual query
        per_token_scores, kv_cache = self.score_per_token_attention_to_query(prompt, query_span, None, 0)

        # Trim kv_cache to pre-query tokens and reset query states for the calibration pass
        self._trim_kv_cache_to(kv_cache, query_span[0])
        self._reset_query_cache(kv_cache, [])
        start_idx = query_span[0]

        null_per_token_scores, _ = self.score_per_token_attention_to_query(null_prompt, null_query_span, kv_cache, start_idx)

        min_length = min(per_token_scores.shape[-1], null_per_token_scores.shape[-1])
        per_token_scores_CAL = per_token_scores[:,:,:min_length] - null_per_token_scores[:,:,:min_length]

        if self.attn_head_set == SPEC_HEAD_SET.FULL_SET:
            per_token_scores_CAL = per_token_scores_CAL.sum(0)
            per_token_scores_CAL = per_token_scores_CAL.sum(0)
        else:
            head_set = self.attn_head_set.split(',')
            head_set = [tuple(map(int, h.split('-'))) for h in head_set]
            indices = torch.tensor(head_set).to(self.device)
            layers = indices[:, 0]
            heads = indices[:, 1]
            per_token_scores_CAL = per_token_scores_CAL[layers, heads]
            per_token_scores_CAL = per_token_scores_CAL.sum(0)

        per_doc_scores = []
        for i, doc_span in enumerate(doc_spans): 
            curr_doc_per_tok_scores_CAL = per_token_scores_CAL[doc_span[0] : doc_span[1]+1]
            threshold = curr_doc_per_tok_scores_CAL.mean() - 2*curr_doc_per_tok_scores_CAL.std()
            tok_mask = (curr_doc_per_tok_scores_CAL > threshold)
            per_doc_scores.append((curr_doc_per_tok_scores_CAL * tok_mask).sum())

        assert len(per_doc_scores) == len(docs)

        results = {}
        for i, doc in enumerate(docs):
            doc_id = doc['idx']
            results[doc_id] = per_doc_scores[i].item()
        return results


    def score_docs_per_head_for_detection(self, query: str, docs: List[Dict]) -> Dict[str, Dict]:
        prompt, tokenized_prompt, query_span, doc_spans = self.compose_scoring_prompt(query, docs)
        null_prompt, tokenized_null_prompt, null_query_span, _ = self.compose_scoring_prompt(self.null_query, docs)

        assert tokenized_null_prompt[:query_span[0]] == tokenized_prompt[:query_span[0]]
        assert query_span[0] == null_query_span[0], "Query start indices do not match between query and null query."

        per_token_scores, kv_cache = self.score_per_token_attention_to_query(prompt, query_span, None, 0)

        # Trim kv_cache to pre-query tokens and reset query states for the calibration pass
        self._trim_kv_cache_to(kv_cache, query_span[0])
        self._reset_query_cache(kv_cache, [])
        start_idx = query_span[0]

        null_per_token_scores, _ = self.score_per_token_attention_to_query(null_prompt, null_query_span, kv_cache, start_idx)

        min_length = min(per_token_scores.shape[-1], null_per_token_scores.shape[-1])
        per_token_scores_CAL = per_token_scores[:,:,:min_length] - null_per_token_scores[:,:,:min_length]

        per_doc_score_tensors = []
        for i, doc_span in enumerate(doc_spans):
            curr_doc_per_tok_scores_CAL = per_token_scores_CAL[:, :, doc_span[0] : doc_span[1]+1]
            threshold = curr_doc_per_tok_scores_CAL.mean(dim=-1) - 2*curr_doc_per_tok_scores_CAL.std(dim=-1)
            tok_mask = curr_doc_per_tok_scores_CAL > threshold.unsqueeze(-1)
            masked_scores = curr_doc_per_tok_scores_CAL.masked_fill(~tok_mask, 0.0)
            masked_scores = masked_scores.sum(dim=-1)
            per_doc_score_tensors.append(masked_scores)

        assert len(per_doc_score_tensors) == len(docs)

        results = {}
        for i, doc in enumerate(docs):
            doc_id = doc['idx']
            results[doc_id] = per_doc_score_tensors[i]
        return results


    def score_per_token_attention_to_query(self, prompt, query_span, kv_cache=None, start_idx=0):
        tokenized_input = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        input_ids = tokenized_input.input_ids[:, start_idx:]
        query_indices = list(range(query_span[0]-start_idx, query_span[1]-start_idx+1))
        
        if kv_cache is None:
            kv_cache = DynamicCacheWithQuery(query_indices=query_indices)
        else:
            self._reset_query_cache(kv_cache, query_indices)

        with torch.no_grad():
            output = self.llm(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=kv_cache,
                output_attentions=True,
                compute_logits=False,
            )
        kv_cache = output.past_key_values
        
        per_token_scores = []
        for i in range(self.start_layer, self.end_layer+1):
            # Access keys and queries directly from the layer objects (5.x API)
            layer = kv_cache.layers[i]
            attn_weights = self._get_attn_weights(layer.keys, layer.queries).to(self.device).squeeze(0)
            attn_weights = attn_weights.mean(1)
            per_token_scores.append(attn_weights.squeeze(0))

        per_token_scores = torch.stack(per_token_scores, dim=0)
        return per_token_scores, kv_cache

    def _get_attn_weights(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.size()
        num_key_value_heads = key_states.size(1)
        num_key_value_groups = num_heads // num_key_value_heads
        kv_seq_len = key_states.size(-2)

        key_states = repeat_kv(key_states, num_key_value_groups)
    
        scale = 1.0 / math.sqrt(head_dim)
        scaled_queries = query_states * scale
        attn_weights = torch.matmul(scaled_queries, key_states.transpose(2,3))

        if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
            raise ValueError(f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is {attn_weights.size()}")
        
        causal_mask = self._get_causal_mask(attn_weights).to(attn_weights.device)
        attn_weights += causal_mask.unsqueeze(0)
        attn_lses = torch.logsumexp(attn_weights, dim=-1, keepdim=True)
        attn_weights = torch.exp(attn_weights - attn_lses)
        return attn_weights
    
    def _get_causal_mask(self, attn_weights):
        query_len, seq_len = attn_weights.size(-2), attn_weights.size(-1)
        causal_mask = torch.ones_like(attn_weights.transpose(-1,-2).squeeze(0))
        causal_mask = torch.triu(causal_mask, diagonal=-(seq_len-query_len))
        causal_mask = causal_mask.transpose(-1,-2)
        causal_mask = (1-causal_mask) * torch.finfo(causal_mask.dtype).min
        return causal_mask


class FullHeadRetriever(AttnBasedRetriever):
    def __init__(self,
        config_or_config_path: Optional[Union[Dict, str]] = None,
        model_name_or_path: Optional[str] = None,
        model_base_class: Optional[str] = None,
        attn_head_set: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__(config_or_config_path, model_name_or_path, model_base_class, attn_head_set, device)
        assert self.attn_head_set == SPEC_HEAD_SET.FULL_SET

    def setup_config(self,
        config_or_config_path: Optional[Union[Dict, str]] = None,
        model_name_or_path: Optional[str] = None,
        model_base_class: Optional[str] = None,
    ):
        if config_or_config_path is not None:
            assert isinstance(config_or_config_path, (str, Dict))
            if isinstance(config_or_config_path, str):
                config = load_config(config_or_config_path)
            else:
                config = config_or_config_path
        else:
            if model_base_class is not None:
                if model_base_class.lower() == 'llama-3.1-8b-instruct':
                    config = load_config(CONFIG_DIR / 'Llama-3.1-8B-Instruct_full_head.yaml')
                elif model_base_class.lower() == 'llama-3.1-70b-instruct':
                    config = load_config(CONFIG_DIR / 'Llama-3.1-70B-Instruct_full_head.yaml')
                elif model_base_class.lower() == 'llama-3.2-3b-instruct':
                    config = load_config(CONFIG_DIR / 'Llama-3.2-3B-Instruct_full_head.yaml')
                elif model_base_class.lower() == 'llama-3.2-1b-instruct':
                    config = load_config(CONFIG_DIR / 'Llama-3.2-1B-Instruct_full_head.yaml')
                elif model_base_class.lower() == 'qwen2.5-7b-instruct':
                    config = load_config(CONFIG_DIR / 'Qwen2.5-7B-Instruct_full_head.yaml')
                else:
                    raise NotImplementedError(f"Config inference for model_base_class {model_base_class} is not implemented.")
            elif model_name_or_path is not None:
                if 'llama-3.1-8b-instruct' in model_name_or_path.lower():
                    config = load_config(CONFIG_DIR / 'Llama-3.1-8B-Instruct_full_head.yaml')
                elif 'llama-3.1-70b-instruct' in model_name_or_path.lower():
                    config = load_config(CONFIG_DIR / 'Llama-3.1-70B-Instruct_full_head.yaml')
                elif 'llama-3.2-3b-instruct' in model_name_or_path.lower():
                    config = load_config(CONFIG_DIR / 'Llama-3.2-3B-Instruct_full_head.yaml')
                elif 'llama-3.2-1b-instruct' in model_name_or_path.lower():
                    config = load_config(CONFIG_DIR / 'Llama-3.2-1B-Instruct_full_head.yaml')
                elif 'qwen2.5-7b-instruct' in model_name_or_path.lower():
                    config = load_config(CONFIG_DIR / 'Qwen2.5-7B-Instruct_full_head.yaml')
                else:
                    raise NotImplementedError(f"Config inference for model_name_or_path {model_name_or_path} is not implemented.")
            else:
                raise ValueError("model_name_or_path or model_base_class is required to use default config")
        return config


class QRRetriever(AttnBasedRetriever):
    def __init__(self,
        config_or_config_path: Optional[Union[Dict, str]] = None,
        model_name_or_path: Optional[str] = None,
        model_base_class: Optional[str] = None,
        attn_head_set: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__(config_or_config_path, model_name_or_path, model_base_class, attn_head_set, device)

    def setup_config(self,
        config_or_config_path: Optional[Union[Dict, str]] = None,
        model_name_or_path: Optional[str] = None,
        model_base_class: Optional[str] = None,
    ):
        if config_or_config_path is not None:
            assert isinstance(config_or_config_path, (str, Dict))
            if isinstance(config_or_config_path, str):
                config = load_config(config_or_config_path)
            else:
                config = config_or_config_path
        else:
            print("config_or_config_path is not provided. Use default config: LME for qr-head.", flush=True)
            if model_base_class is not None:
                if model_base_class.lower() == 'llama-3.1-8b-instruct':
                    config = load_config(CONFIG_DIR / 'Llama-3.1-8B-Instruct_qr_head_LME.yaml')
                elif model_base_class.lower() == 'llama-3.1-70b-instruct':
                    config = load_config(CONFIG_DIR / 'Llama-3.1-70B-Instruct_qr_head_LME.yaml')
                elif model_base_class.lower() == 'llama-3.2-3b-instruct':
                    config = load_config(CONFIG_DIR / 'Llama-3.2-3B-Instruct_qr_head_LME.yaml')
                elif model_base_class.lower() == 'llama-3.2-1b-instruct':
                    config = load_config(CONFIG_DIR / 'Llama-3.2-1B-Instruct_qr_head_LME.yaml')
                elif model_base_class.lower() == 'qwen2.5-7b-instruct':
                    config = load_config(CONFIG_DIR / 'Qwen2.5-7B-Instruct_qr_head_LME.yaml')
                else:
                    raise NotImplementedError(f"Config inference for model_base_class {model_base_class} is not implemented.")
            elif model_name_or_path is not None:
                if 'llama-3.1-8b-instruct' in model_name_or_path.lower():
                    config = load_config(CONFIG_DIR / 'Llama-3.1-8B-Instruct_qr_head_LME.yaml')
                elif 'llama-3.1-70b-instruct' in model_name_or_path.lower():
                    config = load_config(CONFIG_DIR / 'Llama-3.1-70B-Instruct_qr_head_LME.yaml')
                elif 'llama-3.2-3b-instruct' in model_name_or_path.lower():
                    config = load_config(CONFIG_DIR / 'Llama-3.2-3B-Instruct_qr_head_LME.yaml')
                elif 'llama-3.2-1b-instruct' in model_name_or_path.lower():
                    config = load_config(CONFIG_DIR / 'Llama-3.2-1B-Instruct_qr_head_LME.yaml')
                elif 'qwen2.5-7b-instruct' in model_name_or_path.lower():
                    config = load_config(CONFIG_DIR / 'Qwen2.5-7B-Instruct_qr_head_LME.yaml')
                else:
                    raise NotImplementedError(f"Config inference for model_name_or_path {model_name_or_path} is not implemented.")
            else:
                raise ValueError("model_name_or_path or model_base_class is required to use default config")
        return config