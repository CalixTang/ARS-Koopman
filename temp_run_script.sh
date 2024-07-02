#!/bin/sh
conda init
conda activate mjrl-env

# python ARS/ARS/ars.py --params_path params/params_gymnasium.json --policy_type linear --run_name fetchpush-arsl-1 > reports/fetchpush-arsl-1.out
# python ARS/ARS/record_rollout.py --logdir ./data/fetchpush-arsl-1 --num_rollouts 5
python ARS/ARS/ars.py --params_path params/params_gymnasium_tk.json --policy_type minkoopman --run_name fetchpush-arsmk-1 > reports/fetchpush-arsmk-1.out
python ARS/ARS/record_rollout.py --logdir ./data/fetchpush-arstk-1 --num_rollouts 5
python ppo/ppo_koopman.py --params_path params/ppo_params.json --run_name fetchpush-ppotk-1 > reports/fetchpush-ppotk-1.out
python ARS/ARS/record_rollout.py --logdir ./data/fetchpush-ppotk-1 --num_rollouts 5