#!/bin/bash
DIR=`dirname $0`

g++-9 -O0 -std=c++11 -fopenmp -fopenmp-simd -march=native -I"$DIR"/include "$DIR"/SIMD.cpp -o simd_O0
./simd_O0

g++-9 -O3 -std=c++11 -fopenmp -fopenmp-simd -march=native -I"$DIR"/include "$DIR"/SIMD.cpp -o simd_O3
./simd_O3


