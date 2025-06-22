# Retrieval Scripts

This directory contains scripts for running retrieval experiments and evaluating results across multiple datasets.

## Overview

Three datasets we used:
- **BEIR**
- **CLIPPER**
- **LongMemEval**

## Prerequisites

Make sure you have the required data files and configuration files in place:
- Dataset files in their respective directories (`beir_data/`, `clipper_data/`, `longmemeval_data/`)
- Configuration files in `src/qrretriever/configs/`

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

### Clipper Dataset

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