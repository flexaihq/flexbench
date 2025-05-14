from flexbench.tests.server import MODEL_PATH

BASE_CONFIG = {
    "task": "text",
    "model_path": MODEL_PATH,
    "api_server": "http://localhost:1234",
    "dataset_path": "ctuning/MLPerf-OpenOrca",
    "dataset_input_column": "question",
    "dataset_output_column": "response",
    "dataset_system_prompt_column": "system_prompt",
    "total_sample_count": 15,
    "max_generated_tokens": 64,
    "batch_size": 2,
    "target_qps": 5,
}


TEST_CASES = {
    "loadgen-offline-perf": ("loadgen", "Offline", False),
    "loadgen-offline-accuracy": ("loadgen", "Offline", True),
    "loadgen-server-perf": ("loadgen", "Server", False),
    "loadgen-server-accuracy": ("loadgen", "Server", True),
    "vllm-offline-perf": ("vllm", "Offline", False),
    "vllm-server-perf": ("vllm", "Server", False),
    "loadgen-singlestream-perf": ("loadgen", "SingleStream", False),
    "loadgen-singlestream-accuracy": ("loadgen", "SingleStream", True),
}
