#!/bin/bash
cd "$( dirname "${BASH_SOURCE[0]}" )"

./clean_build.sh

cd ..

./build/McVineGPU

echo "Run Finished"
