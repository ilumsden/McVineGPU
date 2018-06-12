# Bash Scripts

This directory contains several Bash scripts that can be used to compile and, in some cases, run the code in this repository. Below are descriptions of each script:
* `build.sh`: compiles the core code (no tests) without replacing any of the up-to-date contents of the `build` directory.
* `clean_build.sh`: clears the `build` directory and then compiles the core code (no tests).
* `run.sh`: compiles the core code like `build.sh` and then runs the `McVineGPU` executable.
* `clean_run.sh`: clears the `build` directory, compiles the core code, and then runs the `McVineGPU` executable.
* `test.sh`: compiles both the core code and the test code without replacing any of the up-to-date contents of the `build` directory.
* `clean_test.sh`: clears the `build` directory and then compiles both the core code and the test code.
* `test_run.sh`: compiles the core code and test code like `test.sh` and then runs the `TestSuite` executable.
* `clean_test_run.sh`: clears the `build` directory, compiles the core and test code, and then runs the `TestSuite` executable.
