# Overview

Three datasets we used:
- **BEIR**
- **CLIPPER**
- **LongMemEval**

## Prerequisites

Make sure you have the required data files and configuration files in place:
- Dataset files in their respective directories (`beir_data/`, `clipper_data/`, `longmemeval_data/`)
- Configuration files in `src/qrretriever/configs/`

# Retrieval Scripts

`retrieval/` directory contains scripts for running retrieval experiments and evaluating results for BEIR, LME and CLIPPER.

## Retrieval Examples

### BEIR Dataset

```bash
python exp_scripts/retrieval/run_retrieval.py \
    --input_file INPUT_DATA_JSON_FILE \
    --output_file RETRIEVAL_RESULT_JSON_FILE \
    --data_type beir \
    --truncate_by_space 400 \
    --retriever_type qr_head \
    --config_or_config_path src/qrretriever/configs/Llama-3.1-8B-Instruct_qr_head_NQ.yaml
```

### CLIPPER Dataset

```bash
python exp_scripts/retrieval/run_retrieval.py \
    --input_file INPUT_DATA_JSON_FILE \
    --output_file RETRIEVAL_RESULT_JSON_FILE \
    --data_type clipper \
    --retriever_type qr_head \
    --config_or_config_path src/qrretriever/configs/Llama-3.1-8B-Instruct_qr_head_LME.yaml
```

### LME Dataset

```bash
python exp_scripts/retrieval/run_retrieval.py \
    --input_file INPUT_DATA_JSON_FILE \
    --output_file RETRIEVAL_RESULT_JSON_FILE \
    --data_type lme \
    --retriever_type qr_head \
    --config_or_config_path src/qrretriever/configs/Llama-3.1-8B-Instruct_qr_head_LME.yaml
```

## Evaluation Examples

After running retrieval, evaluate the results using the evaluation script:

### BEIR Evaluation

```bash
python exp_scripts/retrieval/eval_retrieval_beir.py \
    --retrieval_result_dir RETRIEVAL_RESULT_DIR
```

### Clipper Evaluation

```bash
python exp_scripts/retrieval/eval_retrieval.py \
    --retrieval_result_file RETRIEVAL_RESULT_JSON_FILE \
    --data_file INPUT_DATA_JSON_FILE
```

### LME Evaluation

```bash
python exp_scripts/retrieval/eval_retrieval.py \
    --retrieval_result_file RETRIEVAL_RESULT_JSON_FILE \
    --data_file INPUT_DATA_JSON_FILE
```

# Generation Scripts

`generation/` directory contains scripts for running generation for LME and CLIPPER.

## Generation Examples

### CLIPPER Dataset

```bash
python exp_scripts/generation/run_generation_clipper.py \
    --input_file INPUT_DATA_JSON_FILE \
    --output_file GENERATION_RESULT_JSONL_FILE \
    --model_path MODEL_PATH \
    --retrieval_method retriever \
    --retrieval_result_file RETRIEVAL_RESULT_JSON_FILE \
    --topk_doc 5
```

### LME Dataset

```bash
python exp_scripts/generation/run_generation_lme.py \
    --input_file INPUT_DATA_JSON_FILE \
    --output_file GENERATION_RESULT_JSONL_FILE \
    --model_path MODEL_PATH \
    --retrieval_method retriever \
    --retrieval_result_file RETRIEVAL_RESULT_JSON_FILE \
    --topk_doc 5
```

## Evaluation Examples

After running generation, evaluate the results using the evaluation script:

### Clipper Evaluation

```bash
python exp_scripts/generation/eval_generation_clipper.py \
    --TRUE_generation_result_file GENERATION_RESULT_JSONL_FILE \
    --FALSE_generation_result_file GENERATION_RESULT_JSONL_FILE
```

### LME Evaluation
We use `Llama-3.1-8B-Instruct` model for evluating LME results.

```bash
bash exp_scripts/generation/serve_vllm.sh 0 MODEL_PATH

python exp_scripts/generation/eval_generation_lme.py \
    ----base_url BASE_URL \
    --metric_model MODEL_PATH \
    --generation_result_file GENERATION_RESULT_JSONL_FILE \
    --reference_file DATA_JSON_FILE
```
