#!/bin/bash
DIR=`dirname $0`

g++ -std=c++11 -fopenmp ${DIR}/RC4_par.cpp -I"$DIR"/include/ -o rc4_par
./rc4_par
