#!/usr/bin/env bash

for hist in 100 200 300 400 500;
do
    th benchmarks.lua --history $hist
done