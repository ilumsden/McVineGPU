#!/bin/bash
set -e
set -x

cd "$( dirname "${BASH_SOURCE[0]}" )"
cd ..

rm -rf build
mkdir build
cd build

cmake -DBUILD_TESTING=ON ..
make
