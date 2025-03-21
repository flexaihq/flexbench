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

- HF Datasets: `Open-Orca/OpenOrca`, `ctuning/MLPerf-OpenOrca`, `AI-MO/NuminaMath-TIR`, `ctuning/MLPerf-OpenOrca` (equivalent to `mlcommons-inference-wg-public/open_orca`)
- MLPerf Preprocessed `OpenOrca` pickle (`mlcommons-inference-wg-public/open_orca`): see [official instructions](https://github.com/mlcommons/inference/blob/master/language/llama2-70b/README.md#preprocessed)

To support another HF dataset, you must add the columns mapping (input and output column names) to [DATASET_CONFIGS](dataset.py).

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
- `--task`: Task type, between `text` or `vision`.
- `--scenario`: Choose between `Offline` and `Server`
- `--accuracy`: Accuracy mode (default if not enabled: performance mode)
- `--api-server`: URL of the vLLM API server (default: `http://localhost:8000`)
- `--api-token`: Optional API token for authentication
- `--dataset-path`: Path to dataset on HuggingFace or local pickle file
- `--dataset-split`: Dataset split to use (default: `train`)
- `--dataset-input-column`: Name of the input column in dataset
- `--dataset-output-column`: Name of the output/reference column in dataset
- `--dataset-system-prompt-column`: Optional name of system prompt column
- `--dataset-image-column`: Optional name of image column, in case the task selected is `vision`
- `--total-sample-count`: Number of samples to process
- `--batch-size`: Batch size for offline scenario
- `--max-generated-tokens`: Maximum number of tokens to generate per vLLM request (default: 1024)
- `--tokenizer-path`: Optional HF tokenizer path, in case it is different from the model path (e.g. using a custom model name in vLLM)

<!-- ### 3. Alternative: vLLM Benchmarking (WIP)

For HuggingFace datasets only, you can use vLLM's built-in benchmarking tools.

#### Setup

1. Get and patch vLLM's benchmarking script:

```sh
git clone --branch v0.7.3 --depth 1 https://github.com/vllm-project/vllm.git
cd vllm/ && git apply ../vllm_fix.patch && cd ..
```

Note: the patch only works for a few datasets (`AI-MO/NuminaMath-TIR`, `Open-Orca/OpenOrca`) for now. TODO: create dataset-agnostic patch.

#### Server Mode Benchmark

```sh
python vllm/benchmarks/benchmark_serving.py \
    --model $MODEL_PATH \
    --dataset-name hf \
    --dataset-path $DATASET_PATH \
    --hf-subset default \
    --num-prompts 24576 \
    --request-rate 10
```

#### Offline Mode Benchmark

```sh
CUDA_VISIBLE_DEVICES=0 python vllm/benchmarks/benchmark_throughput.py \
    --model $MODEL_PATH \
    --dataset $DATASET_PATH \
    --hf-subset default \
    --num-prompts 24576
```

Note: The offline benchmark launches its own vLLM instance, so ensure no other vLLM server is running. -->

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
