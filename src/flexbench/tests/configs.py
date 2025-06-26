MODEL_PATH = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Only include parameters that are always valid for all scenarios
BASE_CONFIG = {
    "model_path": MODEL_PATH,
    "api_server": "http://localhost:8000",
    "dataset_path": "ctuning/MLPerf-OpenOrca",
    "dataset_input_column": "question",
    "dataset_output_column": "response",
    "dataset_system_prompt_column": "system_prompt",
    "total_sample_count": 15,
    "max_generated_tokens": 64,
}

# Each test case: (backend, scenario, accuracy, extra_config)
TEST_CASES = {
    # Offline performance: needs target_qps, batch_size
    "loadgen-offline-perf": (
        "loadgen",
        "Offline",
        False,
        {"target_qps": 5, "batch_size": 2},
    ),
    # Offline accuracy: needs accuracy, batch_size, target_qps
    "loadgen-offline-accuracy": (
        "loadgen",
        "Offline",
        True,
        {"batch_size": 2, "target_qps": 5},
    ),
    # Server performance: needs target_qps, no batch_size
    "loadgen-server-perf": (
        "loadgen",
        "Server",
        False,
        {"target_qps": 5},
    ),
    # Server accuracy: needs accuracy, no batch_size, no sweep, target_qps required
    "loadgen-server-accuracy": (
        "loadgen",
        "Server",
        True,
        {"target_qps": 5},
    ),
    # vLLM backend, Offline performance: needs target_qps, batch_size
    "vllm-offline-perf": (
        "vllm",
        "Offline",
        False,
        {"target_qps": 5, "batch_size": 2},
    ),
    # vLLM backend, Server performance: needs target_qps, no batch_size
    "vllm-server-perf": (
        "vllm",
        "Server",
        False,
        {"target_qps": 5},
    ),
    # SingleStream performance: no target_qps, no sweep, batch_size allowed
    "loadgen-singlestream-perf": (
        "loadgen",
        "SingleStream",
        False,
        {"batch_size": 1},
    ),
}
