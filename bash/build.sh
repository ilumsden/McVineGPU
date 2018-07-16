#!/bin/bash
set -e

usage() {
    echo "Usage: $0 [Options]" >&2
    echo
    echo "  Options:"
    echo "  -h|--help   Prints the usage info to the screen"
    echo "  -j#         Specifies the number of threads to use in the \"make\" call."
    echo '                The "#" should be replaced with the number of threads.'
    echo '  -D(Option)  Triggers a debug build by defining the'
    echo '                specified macro.'
    echo '              The following macros (Option) are supported:'
    echo '                - DEBUG:  Full debug build'
    echo '                - PRINT1: Debug build for external intersection.'
    echo '                - PRINT2: Debug build for scattering site calculation.'
    echo '                - PRINT3: Debug build for elastic scattering'
    echo '                          velocity calculation.'
    echo '                - PRINT4: Debug build for internal intersection.'
    exit 1
}

threadcount=10
debugflags=()
for CLA in $@; do
    if [ ${CLA} == $0 ]; then
        continue
    fi
    case "${CLA}" in
        *"-j"*) threadcount=${CLA:2}
        ;;
        *"-D"*) CLA="${CLA}=ON"; debugflags+=(${CLA})
        ;;
        -h|--help) usage
        ;;
        *) ;;
    esac
done

set -x

cd "$( dirname "${BASH_SOURCE[0]}" )"

cd ../build

if [ ${#debugflags[@]} -eq 0 ]; then
    cmake ..
else
    cmake ${debugflags[@]} ..
fi
make -j ${threadcount} VERBOSE=1 &>log.make
