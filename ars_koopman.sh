#!/bin/bash
numbers=(1 2 4 8 16 32)
for i in "${numbers[@]}"
do 
    python ARS/code/ars_koopman.py --deltas_used $i --n_workers 8 > reports/pleaides-arsk-$i.out; 
done

numbers=(64 128 256 320)
for i in "${numbers[@]}"
do 
    python ARS/code/ars.py --deltas_used $i --n_workers 8 > reports/pleaides-ars-$i.out; 
done

    
