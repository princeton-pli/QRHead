# Overview

Three datasets we used:
- **BEIR**
- **CLIPPER**
- **LongMemEval**

## Data
We store preprocessed data for LongMemEval, CLIPPER, and BEIR wihtin this Huggingface repo: [QRHead dataset](https://huggingface.co/datasets/PrincetonPLI/QRHead/tree/main).
* `data/beir_data`
  * `nq_train.json` is used to detect QRHead for BEIR.
  * The remaining data files are used for BEIR evaluation.
* `data/longmemeval_data`
  * `single-session-user_s.json` is used to detect QRHead used for both LongMemEval and CLIPPER.
  * `other_s_original.json` is used for LongMemEval evaluation.
* `data/clipper_data`
  * `test-00000-of-00002.json` is used for CLIPPER evaluation, where the claims are true.
  * `test-00001-of-00002.json` is used for CLIPPER evaluation, where the claims are false.

## Prerequisites

Make sure you have the required data files and configuration files in place:
- Dataset files in their respective directories (`data/beir_data/`, `data/clipper_data/`, `data/longmemeval_data/`)
- Configuration files in `src/qrretriever/configs/`

# QRHead Detection Scripts

`detection/` directory contains scripts for running QRHead detection using `128 samples of BEIR NQ` and `70 samples of LME single-session-user`.

## Detection Examples

### BEIR NQ Dataset

```bash
python exp_scripts/detection/detect_qrhead_beir.py
    --input_file data/beir_data/nq_train.json \
    --output_file DETECTION_RESULT_JSON_FILE \
    --truncate_by_space 400 \
    --config_or_config_path src/qrretriever/configs/Llama-3.1-8B-Instruct_full_head.yaml
```

### LME single-session-user Dataset

```bash
python exp_scripts/detection/detect_qrhead_lme.py
    --input_file data/longmemeval_data/single-session-user_s.json \
    --output_file DETECTION_RESULT_JSON_FILE \
    --config_or_config_path src/qrretriever/configs/Llama-3.1-8B-Instruct_full_head.yaml
```

# Retrieval Scripts

`retrieval/` directory contains scripts for running retrieval and evaluating results for BEIR, LME and CLIPPER.

## Retrieval Examples

### BEIR Dataset

```bash
python exp_scripts/retrieval/run_retrieval.py \
    --input_file data/beir_data/fever_bm25_top_200.json \
    --output_file RETRIEVAL_RESULT_JSON_FILE \
    --data_type beir \
    --truncate_by_space 400 \
    --retriever_type qr_head \
    --config_or_config_path src/qrretriever/configs/Llama-3.1-8B-Instruct_qr_head_NQ.yaml
```

### CLIPPER Dataset

```bash
python exp_scripts/retrieval/run_retrieval.py \
    --input_file data/clipper_data/test-00000-of-00002.json \
    --output_file RETRIEVAL_RESULT_JSON_FILE \
    --data_type clipper \
    --retriever_type qr_head \
    --config_or_config_path src/qrretriever/configs/Llama-3.1-8B-Instruct_qr_head_LME.yaml
```

### LME Dataset

```bash
python exp_scripts/retrieval/run_retrieval.py \
    --input_file data/longmemeval_data/other_s.json \
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
    --data_file data/clipper_data/test-00000-of-00002.json
```

### LME Evaluation

```bash
python exp_scripts/retrieval/eval_retrieval.py \
    --retrieval_result_file RETRIEVAL_RESULT_JSON_FILE \
    --data_file data/longmemeval_data/other_s.json
```

# Generation Scripts

`generation/` directory contains scripts for running and evaluating generation for LME and CLIPPER.

## Generation Examples

### CLIPPER Dataset

```bash
python exp_scripts/generation/run_generation_clipper.py \
    --input_file data/clipper_data/test-00000-of-00002.json \
    --output_file GENERATION_RESULT_JSONL_FILE \
    --model_path MODEL_PATH \
    --retrieval_method retriever \
    --retrieval_result_file RETRIEVAL_RESULT_JSON_FILE \
    --topk_doc 5
```

### LME Dataset

```bash
python exp_scripts/generation/run_generation_lme.py \
    --input_file data/longmemeval_data/other_s_original.json \
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
    --base_url BASE_URL \
    --metric_model MODEL_PATH \
    --generation_result_file GENERATION_RESULT_JSONL_FILE \
    --reference_file data/longmemeval_data/other_s_original.json
```
