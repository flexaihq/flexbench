#!/usr/bin/env bash

# Install vllm 

set -e
set -o pipefail
set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

pushd $SCRIPT_DIR/../
    git clone --depth 1 --branch v0.8.2 --single-branch https://github.com/vllm-project/vllm.git
    pushd vllm
        pip install -r requirements/common.txt
        pip install -r requirements/cpu.txt --index-url https://download.pytorch.org/whl/cpu
        VLLM_TARGET_DEVICE=cpu python setup.py install
    popd
popd