#!/bin/bash
DIR=`dirname $0`

g++-8 -std=c++11 -fopenmp "$DIR"/Find.cpp -o find_omp
./find_omp
