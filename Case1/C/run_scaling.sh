#!/bin/bash
# Collect TSQR timing data for Q3 scaling plots.

BINARY=./tsqr
REPEATS=5

echo "" > results_vary_m.csv
echo "" > results_vary_n.csv

# --- Vary m, fixed n=8 ---
for M in 400 800 1600 3200 6400 12800 25600; do
    for r in $(seq 1 $REPEATS); do
        mpirun -np 4 $BINARY $M 8 >> results_vary_m.csv
    done
done

# --- Vary n, fixed m=4000 ---
for N in 2 4 8 16 32 64 128; do
    for r in $(seq 1 $REPEATS); do
        mpirun -np 4 $BINARY 4000 $N >> results_vary_n.csv
    done
done

echo "Benchmarks done."
