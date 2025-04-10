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
    sudo apt install -y -qq \
        neovim 