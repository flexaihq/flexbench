# FlexBench

## Setup

1. Install uv (optional):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies:

```sh
uv sync
source .venv/bin/activate
```

## Supported Models

Any model from HuggingFace should be supported. Though, specific chat templates are only applied to Llama2 and Llama3 models. Benchmarks were tested on:

- DeepSeek-R1 submodels:
  - Llama distill: `deepseek-ai/DeepSeek-R1-Distill-Llama-70B`, `deepseek-ai/DeepSeek-R1-Distill-Llama-70B`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`, `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
  - Qwen distill: `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- Llama2:
  - Official: `meta-llama/Llama-2-70b-chat-hf`
- Llama3:
  - Official: `meta-llama/Llama-3.1-8B-Instruct`
  - NeuralMagic quantized: `nm-testing/Llama-3.3-70B-Instruct-FP8-dynamic`

Example model configuration:

```sh
export MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # or local pickle
```

## Supported Datasets

### Text Tasks

Any dataset from HuggingFace should be supported, as long as you specify the correct column names for:

- Input text (`--dataset-input-column`)
- Output/reference text (`--dataset-output-column`)
- System prompt (optional, `--dataset-system-prompt-column`)

Common datasets used for benchmarking:

- `ctuning/MLPerf-OpenOrca`
- `Open-Orca/OpenOrca`
- `AI-MO/NuminaMath-TIR`

### Vision Tasks

Currently only supports the `philschmid/amazon-product-descriptions-vlm` dataset (Work in Progress).
Support for additional vision datasets is planned.

Example dataset configuration:

```sh
export DATASET_PATH="ctuning/MLPerf-OpenOrca"  # or local pickle
```

## Running Benchmarks

### 1. Start Model Server (Terminal 1)

Single GPU:

```sh
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_PATH \
    --disable-log-requests \
    --max-model-len=2048
```

Multi-GPU:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve $MODEL_PATH \
    --disable-log-requests \
    --max-model-len=2048 \
    --tensor-parallel-size 4
```

Notes:

- use `--port` arg to change from default (`8000`)
- use `--download-dir` argument to specify vLLM model download directory
- use `--speculative-model` and `--num-speculative-tokens` args to use speculative decoding (if supported by model)

### 2. Run MLPerf Loadgen (Terminal 2)

IMPORTANT: for loadgen to work properly, you must be in the `src/flexbench/` folder:

```sh
cd src/flexbench/
```

Wait for the model server to start, then in a new terminal:

```sh
python main.py \
    --task text \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --api-server http://localhost:8000 \
    --scenario Server \
    --target-qps 10 \
    --dataset-path ctuning/MLPerf-OpenOrca \
    --dataset-input-column question \
    --dataset-output-column response \
    --dataset-system-prompt-column system_prompt \
    --total-sample-count 24576 \
    --accuracy
```

Available arguments:

Required arguments:

- `--task`: Task type (`text` or `vision`)
- `--model-path`: Model name on HuggingFace or local path
- `--api-server`: vLLM API server URL (default: `http://localhost:8000`)
- `--scenario`: MLPerf scenario (`Offline` or `Server`)
- `--target-qps`: Target queries per second
- `--dataset-path`: Dataset path on HuggingFace or local pickle file
- `--dataset-input-column`: Input text column name in dataset

Optional arguments:

- `--accuracy`: Run accuracy evaluation (default: performance mode)  
- `--dataset-output-column`: Reference text column name (required for accuracy mode)
- `--dataset-split`: Dataset split to use (default: `train`)
- `--dataset-system-prompt-column`: System prompt column name
- `--dataset-image-column`: Image column name (required for vision tasks)
- `--tokenizer-path`: Custom tokenizer path if different from model
- `--api-token`: API authentication token
- `--total-sample-count`: Number of samples to process
- `--batch-size`: Batch size for offline scenario
- `--max-generated-tokens`: Max tokens to generate (default: 1024)

## Profiling

To profile the benchmarks:

1. Start the model server with NVIDIA Nsight profiling:

```sh
nsys profile --force-overwrite=true \
    --gpu-metrics-devices=cuda-visible \
    --output=./results/nsys_profiling \
    vllm serve $MODEL_PATH \
    --disable-log-requests \
    --max-model-len=2048
```

2. Run your benchmark normally

3. Stop the server with Ctrl+C when done

4. Generate profiling stats:

```sh
nsys stats --force-overwrite=true \
    --format=table \
    --output=./results/nsys_profiling \
    ./results/nsys_profiling.nsys-rep
```
