#!/usr/bin/env bash

for depth in 1 2 3;
do
    for hist in 100 200 300 400 500;
    do
        th benchmarks.lua --input_size 128 --output_size 128 --history $hist  --depth $depth
    done
done
