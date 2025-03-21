#!/bin/bash

# Terminal 1: Run the VLLM server
CUDA_VISIBLE_DEVICES=0 vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --max-model-len=2048 \
    --disable-log-requests \
    --port 8000

# Terminal 2: Run the FlexBench client
python main.py \
    --task text \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --api-server http://localhost:8000 \
    --scenario Server \
    --target-qps 10 \
    --dataset-path ctuning/MLPerf-OpenOrca \
    --dataset-input-column "question" \
    --dataset-output-column "response" \
    --dataset-system-prompt-column "system_prompt" \
    --total-sample-count 200
