"""Shared configuration builders for FlexBench."""

from flexbench.dataset.base import DatasetConfig
from flexbench.runners.base import BenchmarkConfig


def create_dataset_config(args) -> DatasetConfig:
    """Create DatasetConfig from parsed arguments."""
    return DatasetConfig(
        path=args.dataset_path,
        input_column=args.dataset_input_column,
        output_column=args.dataset_output_column,
        system_prompt_column=args.dataset_system_prompt_column,
        image_column=args.dataset_image_column,
        split=args.dataset_split,
        accuracy_mode=args.accuracy,
    )


def create_benchmark_config(args, dataset_config: DatasetConfig | None = None) -> BenchmarkConfig:
    """Create BenchmarkConfig from parsed arguments."""
    
    if dataset_config is None:
        dataset_config = create_dataset_config(args)
    
    return BenchmarkConfig(
        task=args.task,
        model_path=args.model_path,
        remote_model_path=args.remote_model_path,
        tokenizer_path_override=args.tokenizer_path_override,
        api_server=getattr(args, 'api_server', 'http://localhost:8000'),
        api_token=args.api_token,
        dataset_config=dataset_config,
        scenario=args.scenario,
        target_qps=args.target_qps,
        sweep_mode=args.sweep,
        num_sweep_points=args.num_points,
        batch_size=args.batch_size,
        max_generated_tokens=args.max_generated_tokens,
        max_input_tokens=args.max_input_tokens,
        fixed_input_length=args.fixed_input_length,
        accuracy=args.accuracy,
        total_sample_count=args.total_sample_count,
        output_dir=args.output_dir,
    )
