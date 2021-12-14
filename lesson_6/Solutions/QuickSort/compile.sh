#!/bin/bash
DIR=`dirname $0`

g++-9 -std=c++11 -O0 -fopenmp "$DIR"/QuickSort.cpp -I"$DIR"/include -o quicksort_O0
./quicksort_O0
