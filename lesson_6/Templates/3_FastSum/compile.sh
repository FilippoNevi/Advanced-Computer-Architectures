#!/bin/bash
DIR=`dirname $0`

g++-9 -std=c++11 -fopenmp -I"$DIR"/include "$DIR"/FastSum.cpp -o fast_sum
./fast_sum
