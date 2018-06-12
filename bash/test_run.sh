#!/bin/bash
cd "$( dirname "${BASH_SOURCE[0]}" )"

./test.sh

cd ..

./build/TestSuite

echo "Test Suite Run Finished."
