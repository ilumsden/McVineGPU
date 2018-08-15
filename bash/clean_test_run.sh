#!/bin/bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

if [ $# -gt 0 ]; then
    args=(${@:1})
    ./clean_test.sh ${args}
else
    ./clean_test.sh
fi

cd ..

./build/McVineGPU_test

echo "Test Suite Run Finished."
