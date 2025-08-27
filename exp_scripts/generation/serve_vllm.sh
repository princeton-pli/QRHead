#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;

GPU=$1
MODEL=${2:-meta-llama/Meta-Llama-3.1-8B-Instruct}
PORT=${3:-8000}
TP_SIZE=${4:-1}

export CUDA_VISIBLE_DEVICES=${GPU}

echo "Loading ${MODEL} model..."
python -m vllm.entrypoints.openai.api_server \
       --model ${MODEL} \
       --tensor-parallel-size ${TP_SIZE} \
       --port ${PORT}