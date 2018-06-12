#!/bin/bash
set -e
set -x

cd "$( dirname "${BASH_SOURCE[0]}" )"

cd ../build

cmake ..
make
