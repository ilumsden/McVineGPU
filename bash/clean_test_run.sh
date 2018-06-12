#!/bin/bash
cd "$( dirname "${BASH_SOURCE[0]}" )"

./clean_test.sh

cd ..

./build/TestSuite

echo "Test Suite Run Finished."
