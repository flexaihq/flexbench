# FlexBench

A flexible benchmarking framework for language and vision models, with support for both MLPerf loadgen and vLLM backends.

## Features

- 🚀 Support for both Server (streaming) and Offline (batched) inference modes
- 🔄 Compatible with any HuggingFace model and dataset
- 🎯 MLPerf-compliant benchmarking with loadgen
- 🔍 Performance and accuracy evaluation
- 📊 Detailed metrics including TTFT, throughput, and latency percentiles

## Quick Start

The framework is done to send request to an inference server like Vllm.
In this context to use flexbench, a inference server need to be setup and then flexbench can be use to send and benchmark requests.

  ┌───────────────────┐                       ┌───────────────────┐
  │                   │                       │                   │
  │                   │                       │                   │
  │    Flexbench      │         --->          │       VLLM        │
  │                   │                       │                   │
  │                   │                       │                   │
  └───────────────────┘                       └───────────────────┘

### Environment Setup

-- Flexbench
```sh
./scripts/install_flexbench.sh
```

-- VLLM

On codespace:
```sh
./scripts/install_server_dependencies.sh
./scripts/install_server_python_dependencies.sh
./scripts/install_server_vllm.sh
```

On MacOS:
```sh
./scripts/install_macos_server_dependencies.sh
./scripts/install_server_python_dependencies.sh
./scripts/install_server_vllm.sh
```


### Prerequisites

1. Install uv (recommended):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Option 1: Remote Endpoint

If you're using a remote API endpoint, you only need the client:

```sh
uv sync
uv pip install -e .
```

### Option 2: Local Deployment

If you want to run the model locally with vLLM:

```sh
uv sync
uv pip install -e ".[local]"
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
```

# Multi-GPU
```sh
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

## Next steps

- Push results to [FlexBoard](https://github.com/flexaihq/flexboard).
- Collect and record all relevant information about hardware, software, model, dataset, and benchmarking results for further data analytics and predictive modeling.
- Compare results with existing MLPerf data.
- Add telemetry with nsys and sampling using hardware counters.
- Build predictive models to suggest the most optimal hardware for running a given model, and integrate them with FCS.

## Project page

- https://www.notion.so/flexaihq/FCS-Labs-2025-14aec14ca14580f793d1d82ee7c409fc?pvs=4
- https://www.notion.so/flexaihq/FlexBoard-181ec14ca14580168baed2d601eedb14?pvs=4

## Authors

Daniel Altunay and Grigori Fursin (FCS Labs)
