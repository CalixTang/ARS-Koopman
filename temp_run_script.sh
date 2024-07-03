#!/bin/sh
conda init
conda activate mjrl-env

# python ARS/ARS/ars.py --params_path params/params_gymnasium.json --policy_type linear --run_name fetchpush-arsl-1 > reports/fetchpush-arsl-1.out
# python ARS/ARS/record_rollout.py --logdir ./data/fetchpush-arsl-1 --num_rollouts 5
# python ARS/ARS/ars.py --params_path params/params_gymnasium_tk.json --policy_type minkoopman --run_name fetchpush-arsmk-1 > reports/fetchpush-arsmk-1.out
# python ARS/ARS/record_rollout.py --logdir ./data/fetchpush-arstk-1 --num_rollouts 5
python ppo/ppo_koopman.py --params_path params/ppo_params.json --run_name fetchpush-ppomk-1 --ppo_lr .0001 > reports/fetchpush-ppomk-1.out
python ARS/ARS/record_rollout.py --logdir ./data/fetchpush-ppomk-1 --num_rollouts 5
python ppo/ppo_koopman.py --params_path params/ppo_params.json --run_name fetchpush-ppomk-2 --ppo_lr .0005 > reports/fetchpush-ppomk-2.out
python ARS/ARS/record_rollout.py --logdir ./data/fetchpush-ppomk-2 --num_rollouts 5
python ppo/ppo_koopman.py --params_path params/ppo_params.json --run_name fetchpush-ppomk-3 --ppo_lr .001 > reports/fetchpush-ppomk-3.out
python ARS/ARS/record_rollout.py --logdir ./data/fetchpush-ppomk-3 --num_rollouts 5