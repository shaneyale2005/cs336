#!/bin/bash

# Get the Python executable path from the uv environment
PYTHON_EXEC=$(uv run which python)

# Get pybind11 include path using the uv-managed python
PYBIND11_INCLUDES=$("$PYTHON_EXEC" -m pybind11 --includes)

# Get the correct extension suffix using the uv-managed python
EXT_SUFFIX=$("$PYTHON_EXEC" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Build bpe.cxx
c++ -I$(pwd) -O3 -Wall -shared -std=c++17 -fPIC $PYBIND11_INCLUDES bpe.cxx -o bpe$EXT_SUFFIX

# Build encoder.cxx
c++ -I$(pwd) -O3 -Wall -shared -std=c++17 -fPIC $PYBIND11_INCLUDES encoder.cxx -o encoder$EXT_SUFFIX