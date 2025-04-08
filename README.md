# FlexBench

A flexible benchmarking framework for language and vision models, with support for both MLPerf loadgen and vLLM backends.

## Features

- 🚀 Support for both Server (streaming) and Offline (batched) inference modes
- 🔄 Compatible with any HuggingFace model and dataset
- 🎯 MLPerf-compliant benchmarking with loadgen
- 🔍 Performance and accuracy evaluation
- 📊 Detailed metrics including TTFT, throughput, and latency percentiles

## Quick Start

1. Install uv (recommended):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies and project:

```sh
uv sync
uv pip install -e .
# Install vLLM
uv pip install -e .[local]
```

3. Set up virtual environment:

```sh
source .venv/bin/activate
```

## Model Support

FlexBench works with any HuggingFace model, with specialized chat templates for:

- Llama2 models (`meta-llama/Llama-2-*`)
- Llama3 models (`meta-llama/Llama-3-*`)
- DeepSeek models (`deepseek-ai/DeepSeek-*`)

### Tested Models

- **DeepSeek-R1 Series**
  - Large: `deepseek-ai/DeepSeek-R1-Distill-Llama-70B`
  - Medium: `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
  - Small: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- **Llama Official**
  - Llama2: `meta-llama/Llama-2-70b-chat-hf`
  - Llama3: `meta-llama/Llama-3.1-8B-Instruct`
  - Quantized: `nm-testing/Llama-3.3-70B-Instruct-FP8-dynamic`

## Dataset Support

### Text Tasks

FlexBench supports any HuggingFace dataset with configurable column mapping for:

- Input text (`--dataset-input-column`), required
- Output/reference text (`--dataset-output-column`), for accuracy mode only
- System prompt (`--dataset-system-prompt-column`), optional

Commonly used datasets:

- `ctuning/MLPerf-OpenOrca`
- `Open-Orca/OpenOrca`
- `AI-MO/NuminaMath-TIR`

### Vision Tasks

Currently supports:

- `philschmid/amazon-product-descriptions-vlm` (Beta)

## Usage Examples

### 1. Server Mode (Streaming)

First, start the vLLM server:

```sh
# Single GPU
CUDA_VISIBLE_DEVICES=0 vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --disable-log-requests \
    --max-model-len=2048

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --disable-log-requests \
    --max-model-len=2048 \
    --tensor-parallel-size 4
```

Then run the benchmark:

```sh
python -m flexbench \
    --task text \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --api-server http://localhost:8000 \
    --scenario Server \
    --target-qps 10 \
    --dataset-path ctuning/MLPerf-OpenOrca \
    --dataset-input-column question \
    --dataset-output-column response \
    --dataset-system-prompt-column system_prompt \
    --total-sample-count 200
```

### 2. Offline Mode (Batched)

```sh
python -m flexbench \
    --task text \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --api-server http://localhost:8000 \
    --scenario Offline \
    --batch-size 32 \
    --target-qps inf \
    --dataset-path ctuning/MLPerf-OpenOrca \
    --dataset-input-column question \
    --total-sample-count 200
```

### 3. Remote Endpoint

```sh
python -m flexbench \
    --task text \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --api-server https://your-endpoint.com \
    --api-token your_token \
    --scenario Server \
    --target-qps 10 \
    --dataset-path ctuning/MLPerf-OpenOrca \
    --dataset-input-column question
```

## Advanced Usage

### Benchmark Backends

FlexBench supports multiple backend implementations:

1. **MLPerf LoadGen** (default)

   - MLPerf-compliant benchmarking
   - Supports both performance and accuracy modes
   - Example:

   ```sh
   python -m flexbench \
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

2. **vLLM Direct**
   - Native vLLM streaming support
   - Simpler implementation without MLPerf overhead
   - Adapted from [vllm/benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks)
   - Example:
   ```sh
   python -m flexbench \
       --task text \
       --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
       --api-server http://localhost:8000 \
       --scenario Server \
       --target-qps 10 \
       --dataset-path ctuning/MLPerf-OpenOrca \
       --dataset-input-column question \
       --backend vllm
   ```

### Profiling with NVIDIA Nsight

1. Profile the server:

```sh
nsys profile --force-overwrite=true \
    --gpu-metrics-devices=cuda-visible \
    --output=./results/nsys_profiling \
    vllm serve $MODEL_PATH \
    --disable-log-requests \
    --max-model-len=2048
```

2. Generate stats:

```sh
nsys stats --force-overwrite=true \
    --format=table \
    --output=./results/nsys_profiling \
    ./results/nsys_profiling.nsys-rep
```

## Running Tests

Tests are located in `src/flexbench/tests/` and use SmolLM2-135M with MLPerf-OpenOrca dataset.

Run them with:

```sh
pytest . -v -s
```

The tests will automatically:

1. Start a vLLM server with the test model
2. Run all test cases
3. Shut down the server when done

The test suite covers:

- vLLM backend (Server and Offline modes)
- LoadGen backend (Server and Offline modes, both performance and accuracy tests)

The tests use minimal samples and a small model for quick validation.
