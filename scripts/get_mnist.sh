#!/usr/bin/env bash
# Download MNIST into ./data (uncompressed idx files).
set -euo pipefail
DIR="${1:-data}"
BASE="https://ossci-datasets.s3.amazonaws.com/mnist"
mkdir -p "$DIR"
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  if [ ! -f "$DIR/$f" ]; then
    echo "downloading $f"
    curl -fsSL "$BASE/$f.gz" -o "$DIR/$f.gz"
    gunzip -f "$DIR/$f.gz"
  fi
done
ls -la "$DIR"
