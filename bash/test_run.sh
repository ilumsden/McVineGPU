#!/bin/bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

if [ $# -gt 0 ]; then
    args=(${@:1})
    ./test.sh ${args}
else
    ./test.sh
fi

cd ..

./build/McVineGPU_Test

echo "Test Suite Run Finished."
