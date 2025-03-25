#!/bin/bash

# Define particle counts to iterate over
particles=(1000 10000 100000 1000000 5000000 10000000)

# Run ./gpu for each particle count
for n in "${particles[@]}"; do
    echo "Running ./gpu with -n $n -s 1"
    ./gpu -n "$n" -s 1
done
