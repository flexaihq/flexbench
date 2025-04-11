#!/usr/bin/env bash

# Install vllm 

set -e
set -o pipefail
set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

pushd $SCRIPT_DIR/../
    pip install -e .
popd
