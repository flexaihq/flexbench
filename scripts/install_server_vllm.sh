#!/usr/bin/env bash

# Install vllm 

set -e
set -o pipefail
set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
pushd $SCRIPT_DIR/../
    git submodule sync
    git submodule update --init
    pushd vendor/vllm
        pip install -r requirements/common.txt
        pip install -r requirements/cpu.txt --index-url https://download.pytorch.org/whl/cpu
        VLLM_CPU_DISABLE_AVX512=${VLLM_CPU_DISABLE_AVX512:=true} VLLM_TARGET_DEVICE=cpu python setup.py install
    popd
popd
