#!/bin/bash
cd "$( dirname "${BASH_SOURCE[0]}" )"

./build.sh

cd ..

./build/McVineGPU

echo "Run Finished"
