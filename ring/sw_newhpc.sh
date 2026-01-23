#!/bin/bash

j=1
while [ $j -le 300 ]
do
    echo "Iteration $j"

    gpumd || exit 1

    rm -f model.xyz
    python3 sw_transfrom.py restart.xyz model.xyz || exit 1
    rm -f restart.xyz

    ((j++))
done
