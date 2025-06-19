#!/usr/bin/env python3
"""Comprehensive test for device type implementation."""

import sys
sys.path.insert(0, 'src')

from flexbench.args import create_cli_parser, validate_args
from flexbench.cli.config import create_docker_config_from_args


def test_edge_cases():
    """Test edge cases and error conditions."""
    parser = create_cli_parser()
    
    print("=== Testing Edge Cases ===")
    
    # Test 1: Default configuration (should be NVIDIA)
    print("\n1. Testing default device type:")
    args = parser.parse_args([
        "--task", "text",
        "--model-path", "test-model",
        "--scenario", "Server",
        "--target-qps", "10",
        "--dataset-path", "test-dataset", 
        "--dataset-input-column", "question"
    ])
    args = validate_args(args)
    config = create_docker_config_from_args(args)
    
    assert config.docker_config.device_type == "nvidia"
    assert not config.docker_config.needs_build_from_source
    print("✓ Default device type is nvidia")
    
    # Test 2: ROCm with build args parsing
    print("\n2. Testing ROCm build args parsing:")
    args = parser.parse_args([
        "--task", "text",
        "--model-path", "test-model", 
        "--scenario", "Server",
        "--target-qps", "10",
        "--dataset-path", "test-dataset",
        "--dataset-input-column", "question",
        "--device-type", "rocm",
        "--vllm-build-args", "PYTORCH_ROCM_ARCH=gfx1100 VLLM_TARGET_DEVICE=rocm"
    ])
    args = validate_args(args)
    config = create_docker_config_from_args(args)
    
    assert config.docker_config.device_type == "rocm"
    assert config.docker_config.needs_build_from_source
    assert config.docker_config.vllm_build_args["PYTORCH_ROCM_ARCH"] == "gfx1100"
    assert config.docker_config.vllm_build_args["VLLM_TARGET_DEVICE"] == "rocm"
    print("✓ ROCm build args parsed correctly")
    
    # Test 3: CPU with no GPU settings
    print("\n3. Testing CPU device type:")
    args = parser.parse_args([
        "--task", "text",
        "--model-path", "test-model",
        "--scenario", "Server", 
        "--target-qps", "5",
        "--dataset-path", "test-dataset",
        "--dataset-input-column", "question",
        "--device-type", "cpu"
    ])
    args = validate_args(args)
    config = create_docker_config_from_args(args)
    
    assert config.docker_config.device_type == "cpu"
    assert config.docker_config.needs_build_from_source
    assert config.docker_config.gpu_devices is None
    assert config.docker_config.gpu_count is None
    print("✓ CPU device type configured correctly")
    
    # Test 4: Custom vLLM repo and branch
    print("\n4. Testing custom vLLM repo:")
    args = parser.parse_args([
        "--task", "text",
        "--model-path", "test-model",
        "--scenario", "Server",
        "--target-qps", "10", 
        "--dataset-path", "test-dataset",
        "--dataset-input-column", "question",
        "--device-type", "cpu",
        "--vllm-repo", "https://github.com/custom/vllm.git",
        "--vllm-branch", "v0.3.0"
    ])
    args = validate_args(args)
    config = create_docker_config_from_args(args)
    
    assert config.docker_config.vllm_repo == "https://github.com/custom/vllm.git"
    assert config.docker_config.vllm_branch == "v0.3.0"
    print("✓ Custom vLLM repo and branch set correctly")
    
    print("\n=== All tests passed! ===")


def test_dockerfile_mappings():
    """Test that dockerfile mappings are correct."""
    from flexbench.cli.config import DockerConfig
    
    print("\n=== Testing Dockerfile Mappings ===")
    
    config = DockerConfig(device_type="nvidia")
    assert config.vllm_dockerfile == "Dockerfile"
    assert config.custom_vllm_image_name == "vllm-nvidia:latest"
    print("✓ NVIDIA dockerfile mapping correct")
    
    config = DockerConfig(device_type="cpu")
    assert config.vllm_dockerfile == "Dockerfile.cpu"
    assert config.custom_vllm_image_name == "vllm-cpu:latest"
    print("✓ CPU dockerfile mapping correct")
    
    config = DockerConfig(device_type="rocm")
    assert config.vllm_dockerfile == "Dockerfile.rocm"
    assert config.custom_vllm_image_name == "vllm-rocm:latest"
    print("✓ ROCm dockerfile mapping correct")


if __name__ == "__main__":
    test_edge_cases()
    test_dockerfile_mappings()
    print("\n🎉 All comprehensive tests passed!")
