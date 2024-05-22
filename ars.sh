#!/bin/bash
numbers=(1 2 4 8 16 32)
for i in "${numbers[@]}"
do 
    python ARS/code/ars.py --deltas_used $i --n_workers 8 > reports/pleaides-ars-$i.out;
done

