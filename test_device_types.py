#!/usr/bin/env python3
"""Test script to verify device type configuration works correctly."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from flexbench.args import create_cli_parser, validate_args
from flexbench.cli.config import create_docker_config_from_args

def test_device_configs():
    """Test different device type configurations."""
    
    # Create parser
    parser = create_cli_parser()
    
    # Test NVIDIA configuration (default)
    print("=== Testing NVIDIA Configuration ===")
    args = parser.parse_args([
        "--task", "text",
        "--model-path", "HuggingFaceTB/SmolLM2-135M-Instruct",
        "--scenario", "Server",
        "--target-qps", "10",
        "--dataset-path", "ctuning/MLPerf-OpenOrca",
        "--dataset-input-column", "question",
        "--gpu-devices", "0,1"
    ])
    args = validate_args(args)
    config = create_docker_config_from_args(args)
    
    print(f"Device type: {config.docker_config.device_type}")
    print(f"Needs build from source: {config.docker_config.needs_build_from_source}")
    print(f"vLLM image: {config.docker_config.vllm_image}")
    print(f"Dockerfile: {config.docker_config.vllm_dockerfile}")
    print()
    
    # Test CPU configuration
    print("=== Testing CPU Configuration ===")
    args = parser.parse_args([
        "--task", "text",
        "--model-path", "HuggingFaceTB/SmolLM2-135M-Instruct",
        "--scenario", "Server",
        "--target-qps", "5",
        "--dataset-path", "ctuning/MLPerf-OpenOrca",
        "--dataset-input-column", "question",
        "--device-type", "cpu"
    ])
    args = validate_args(args)
    config = create_docker_config_from_args(args)
    
    print(f"Device type: {config.docker_config.device_type}")
    print(f"Needs build from source: {config.docker_config.needs_build_from_source}")
    print(f"Custom vLLM image: {config.docker_config.custom_vllm_image_name}")
    print(f"Dockerfile: {config.docker_config.vllm_dockerfile}")
    print()
    
    # Test ROCm configuration
    print("=== Testing ROCm Configuration ===")
    args = parser.parse_args([
        "--task", "text",
        "--model-path", "HuggingFaceTB/SmolLM2-135M-Instruct",
        "--scenario", "Server",
        "--target-qps", "8",
        "--dataset-path", "ctuning/MLPerf-OpenOrca",
        "--dataset-input-column", "question",
        "--device-type", "rocm",
        "--gpu-devices", "0",
        "--vllm-build-args", "PYTORCH_ROCM_ARCH=gfx1100"
    ])
    args = validate_args(args)
    config = create_docker_config_from_args(args)
    
    print(f"Device type: {config.docker_config.device_type}")
    print(f"Needs build from source: {config.docker_config.needs_build_from_source}")
    print(f"Custom vLLM image: {config.docker_config.custom_vllm_image_name}")
    print(f"Dockerfile: {config.docker_config.vllm_dockerfile}")
    print(f"Build args: {config.docker_config.vllm_build_args}")
    print()

if __name__ == "__main__":
    test_device_configs()
