#!/bin/bash
DIR=`dirname $0`

g++-9 -O0 -std=c++11 -fopenmp -I"$DIR"/include "$DIR"/Fibonacci.cpp -o fibonacci_O0
./fibonacci_O0

#g++-9 -O3 -std=c++11 -fopenmp -I"$DIR"/include "$DIR"/Fibonacci.cpp -o fibonacci_O3
#./fibonacci_O3
