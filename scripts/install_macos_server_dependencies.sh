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

brew install gcc@12
check_installed vim

set +x
