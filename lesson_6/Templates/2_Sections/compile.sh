#!/bin/bash
DIR=`dirname $0`

g++-9 -std=c++11 -fopenmp "$DIR"/Sections.cpp -o sections
./sections
