#!/bin/bash
set -e

usage() {
    echo "Usage: $0 [-h|--help] [-j#]" >&2
    echo "    -h|--help    Prints the usage info to the screen"
    echo "    -j#         Specifies the number of threads to use in the \"make\" call." 
    echo '                The "#" should be replaced with the number of threads.'
    exit 1
}

threadcount=10
for CLA in $@; do
    if [ ${CLA} == $0 ]; then
        continue
    fi
    case "${CLA}" in
        *"-j"*) threadcount=${CLA:2}
        ;;
        -h|--help) usage
        ;;
        *) ;;
    esac
done

set -x

cd "$( dirname "${BASH_SOURCE[0]}" )"
cd ..

cd build

cmake -DBUILD_TESTING=ON ..
make -j ${threadcount} VERBOSE=1 &>log.make
