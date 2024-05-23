#!/bin/bash
numbers=(1 5 10 25 50 75 100 200 300 500)
for i in "${numbers[@]}"
do 
    python ARS/code/ars_koopman.py --policy_type eigenrelocate --num_modes $i --n_workers 8 > reports/pleaides-arsek-$i.out; 
done

# numbers=(64 128 256 320)
# for i in "${numbers[@]}"
# do 
#     python ARS/code/ars.py --deltas_used $i --n_workers 8 > reports/pleaides-ars-$i.out; 
# done

    
