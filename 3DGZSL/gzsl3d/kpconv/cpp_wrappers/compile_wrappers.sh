#!/bin/bash

# Compile cpp subsampling
cd cpp_subsampling
python setup.py build_ext --inplace
cd ..

# Compile cpp radius_neighbors
cd cpp_neighbors
python setup.py build_ext --inplace
cd ..
