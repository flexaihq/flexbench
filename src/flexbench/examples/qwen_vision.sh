#!/bin/bash

# Terminal 1: Run the VLLM server
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --max-model-len=2048 \
    --disable-log-requests \
    --port 8000

# Terminal 2: Run the FlexBench client
python main.py \
    --task vision \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --api-server http://localhost:8000 \
    --scenario Server \
    --target-qps 10 \
    --dataset-path philschmid/amazon-product-descriptions-vlm \
    --dataset-input-column "Product Name" \
    --dataset-output-column "description" \
    --dataset-image-column "image" \
    --total-sample-count 200
