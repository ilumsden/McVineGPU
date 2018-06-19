#!/bin/bash

cd "$( dirname "${BASH_SOURCE[0]}" )"
if [ $# -gt 0 ]; then
    args=(${@:1})
    ./clean_build.sh ${args}
else
    ./clean_build.sh
fi

cd ..

./build/McVineGPU

echo "Run Finished"
