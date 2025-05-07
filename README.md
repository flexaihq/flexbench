# FlexBench

A flexible benchmarking framework for language and vision (WIP) models, with support for both MLPerf loadgen and vLLM backends.

## Features

- 🚀 Support for both Server (streaming) and Offline (batched) inference modes
- 🔄 Compatible with any HuggingFace model and dataset
- 🎯 MLPerf-compliant benchmarking with loadgen
- 🔍 Performance and accuracy evaluation
- 📊 Detailed metrics including TTFT, throughput, and latency percentiles
- 📈 QPS sweep mode for discovering performance characteristics

## Architecture

FlexBench uses a client-server architecture where the client (FlexBench) connects to a running vLLM server:

```mermaid
flowchart LR
    subgraph Terminal 1
    B[vLLM Server] -->|Model Inference| C[HuggingFace Model]
    end
    subgraph Terminal 2
    A[FlexBench Client] <-->|API Requests/Responses| B
    A -->|Loads Dataset| D[HuggingFace Dataset]
    A -->|Records Metrics| E[Results]
    end
```

**Important:** The vLLM server and FlexBench client run in separate terminals. You must start the vLLM server first, then run FlexBench in another terminal.

## Inference Modes

FlexBench supports two primary inference modes based on MLPerf standards:

### Server Mode (Streaming)

In Server mode, queries arrive at the system following a Poisson distribution, mimicking real-world request patterns:

![Server Mode Query Pattern](./assets/server_mode_chart.png)

### Offline Mode (Batched)

In Offline mode, all queries are sent to the system at once, maximizing throughput:

![Offline Mode Query Pattern](./assets/offline_mode_chart.png)

### MLPerf Inference Benchmark

For more details on the MLPerf Inference Benchmark and the design of modes and metrics, refer to the [MLPerf Inference Benchmark paper](https://arxiv.org/pdf/1911.02549).

## Quick Start

### Installation

Using uv (recommended, more stable):

```sh
# Optional: install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup environment
uv sync
source .venv/bin/activate
uv pip install -e .              # Basic installation for remote endpoints
# uv pip install -e ".[local]"   # For local model inference deployment
```

Using standard venv:

```sh
# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -e .                 # Basic installation for remote endpoints
# pip install -e ".[local]"      # For local model inference deployment
```

The `[local]` option installs additional dependencies required for local model inference with vLLM.

## Model & Dataset Support

### Models

FlexBench works with any HuggingFace model, with specialized chat templates for:

- Llama2 models (`meta-llama/Llama-2-*`)
- Llama3 models (`meta-llama/Llama-3-*`)
- DeepSeek models (`deepseek-ai/DeepSeek-*`)

### Dataset Support

#### Text Tasks

- Configurable column mapping for input text, output text, and system prompts
- Examples: `ctuning/MLPerf-OpenOrca`, `Open-Orca/OpenOrca`

#### Vision Tasks

- Support for `philschmid/amazon-product-descriptions-vlm` (prototype, WIP)

## Usage

### Terminal 1: Start a vLLM Server

First, start the vLLM server in one terminal:

Single GPU:

```sh
vllm serve HuggingFaceTB/SmolLM2-135M-Instruct --disable-log-requests --max-model-len=2048
```

Multi-GPU:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve HuggingFaceTB/SmolLM2-135M-Instruct --disable-log-requests --max-model-len=2048 --tensor-parallel-size 4
```

### Terminal 2: Run FlexBench

WARNING: ensure the vLLM server is running before executing FlexBench

In a second terminal, run FlexBench to connect to the vLLM server:

```sh
# Standard mode with target QPS
python -m flexbench \
    --task text \
    --model-path HuggingFaceTB/SmolLM2-135M-Instruct \
    --api-server http://localhost:8000 \
    --scenario Server \
    --target-qps 10 \
    --dataset-path ctuning/MLPerf-OpenOrca \
    --dataset-input-column question \
    --dataset-system-prompt-column system_prompt \
    --total-sample-count 100
```

For sweep mode to discover performance characteristics:

```sh
python -m flexbench \
    --task text \
    --model-path HuggingFaceTB/SmolLM2-135M-Instruct \
    --api-server http://localhost:8000 \
    --scenario Server \
    --sweep \
    --num-points 20 \
    --dataset-path ctuning/MLPerf-OpenOrca \
    --dataset-input-column question \
    --dataset-system-prompt-column system_prompt \
    --total-sample-count 100
```

Note: use `LOG_LEVEL=DEBUG` env variable to enable debug logging.

## Key Parameters

| Parameter | Description | Available Options |
|-----------|-------------|-------------------|
| `--task` | Task type | `text`, `vision` (WIP) |
| `--scenario` | MLPerf scenario | `Server` (streaming), `Offline` (batched) |
| `--backend` | Benchmark implementation | `loadgen` (MLPerf-compliant), `vllm` (direct - *WIP*) |
| `--accuracy` | Evaluation mode | Flag to enable accuracy mode (default: performance). Needs `--dataset-output-column` to be set. Not compatible with `--sweep`. |
| `--dataset-output-column` | Reference text column (for accuracy mode) | String |
| `--target-qps` | Target query rate to achieve | Float |
| `--sweep` | Sweep mode | Flag to enable QPS sweep mode (incompatible with `--target-qps` and `--accuracy`). Automatically tests multiple QPS levels to discover performance limits and saturation points. |
| `--num-points` | Number of QPS points in sweep | Integer (default: 10) |
| `--batch-size` | Batch size, for Offline mode only | Integer |
| `--max-input-tokens` | Maximum number of tokens for input | Integer (longer inputs will be truncated) |
| `--fixed-input-length` | Fixed input length flag | Flag to pad inputs to exactly `--max-input-tokens` length (padding on right side) |

For more details on each parameter, use `python -m flexbench --help`.

### Sweep Mode

Sweep mode automates the process of finding your model's performance curve by:

1. First determining the maximum throughput your model can handle
2. Then testing a range of QPS values (from low to high) to map the complete performance profile
3. Capturing metrics like latency and throughput at each level to identify optimal operating points

This is useful for capacity planning and understanding how your model performs under various load conditions. Results include comprehensive metrics at each tested QPS level.

### Offline Mode Batching Behavior

In Offline mode, the `--batch-size` parameter controls query processing:

- **Default (no value specified)**: All samples processed as a single batch for maximum throughput
- **Custom value**: All queries are still received at once, but processed in smaller chunks:
  - Queries divided into batches of specified size
  - Multiple worker threads process these batches in parallel
  - Each batch becomes a separate API call to the inference server

This maintains MLPerf methodology (submitting all queries at the start) while allowing flexible processing.

## Additional Options

For dataset configuration:

- `--dataset-input-column`: Input text column (required)
- `--dataset-output-column`: Reference text column (for accuracy mode)
- `--dataset-system-prompt-column`: System prompt column (optional)
- `--dataset-image-column`: Image column (for vision tasks, WIP)

For API configuration:

- `--api-server`: vLLM server URL
- `--api-token`: Authentication token for remote endpoints

## Testing & Development

Run tests with:

```sh
pytest . -v -s
```

## Authors

Daniel Altunay and Grigori Fursin (FCS Labs)
