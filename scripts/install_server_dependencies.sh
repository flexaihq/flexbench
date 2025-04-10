#!/usr/bin/env bash

set -e
set -o pipefail
set -x


check_installed() {
    echo "Checking for '$1'..."
    if ! command -v "$1" &>/dev/null; then
        echo "Error: '$1' is not installed."
        exit 1
    fi
}

# Check prerequisites
check_installed git
check_installed pip

echo "Installing prerequisites..."

DEBIAN_FRONTEND=noninteractive \
    sudo apt update -qq &&
    sudo apt install -y -qq gcc g++ libnuma-dev 

check_installed vim

# Install dependencies
echo "Installing Python dependencies..."
python -m pip install --upgrade pip
pip install "cmake>=3.26" wheel packaging ninja "setuptools-scm>=8" numpy datasets
pip install intel_extension_for_pytorch==2.6.0

set +x