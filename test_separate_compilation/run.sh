#!/bin/bash

echo "mylib.o"
nvcc -g -G -dc -o mylib.o -c mylib.cu
echo "test.o"
nvcc -g -G -dc -o test.o -c test.cu
echo "mylib.dlink.o"
nvcc -dlink -o mylib.dlink.o mylib.o test.o
#echo "test.dlink.o"
#nvcc -dlink -o test.dlink.o test.o
echo "Making library"
nvcc -lib -o mylib.a mylib.dlink.o mylib.o test.o
echo "Compiling main."
g++ -I/usr/local/cuda/include -o main.o -c main.cpp
echo "Linking executable"
g++ main.o mylib.a -o test -L/usr/local/cuda/lib64 -lcudart -lcudadevrt

./test
