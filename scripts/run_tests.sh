#!/usr/bin/env bash

# Install test dependencies 

set -e
set -o pipefail
set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

pushd $SCRIPT_DIR/../
    pytest -v -s
popd