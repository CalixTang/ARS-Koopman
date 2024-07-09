#!/bin/sh
# conda init
# conda activate mjrl-env

# python ARS/ARS/ars.py --params_path params/params_gymnasium.json --policy_type linear --run_name fetchpush-arsl-1 > reports/fetchpush-arsl-1.out
# python ARS/ARS/record_rollout.py --logdir ./data/fetchpush-arsl-1 --num_rollouts 5
# python ARS/ARS/ars.py --params_path params/params_gymnasium_tk.json --policy_type minkoopman --run_name fetchpush-arsmk-1 > reports/fetchpush-arsmk-1.out
# python ARS/ARS/record_rollout.py --logdir ./data/fetchpush-arstk-1 --num_rollouts 5
# python ppo/ppo_koopman.py --params_path params/ppo_params.json --run_name fetchpush-ppomk-1 --actor_noise 0.04 0.04 0.04 0.00 > reports/fetchpush-ppomk-1.out
# python ARS/ARS/record_rollout.py --logdir ./data/fetchpush-ppomk-1 --num_rollouts 5
# python ppo/ppo_koopman.py --params_path params/ppo_params.json --run_name fetchpush-ppomk-2 --actor_noise 0.1 0.1 0.1 0.00 > reports/fetchpush-ppomk-2.out
# python ARS/ARS/record_rollout.py --logdir ./data/fetchpush-ppomk-2 --num_rollouts 5
# python ppo/ppo_koopman.py --params_path params/ppo_params.json --run_name fetchpush-ppomk-3 --actor_noise 0.01 0.01 0.01 0.00 > reports/fetchpush-ppomk-3.out
# python ARS/ARS/record_rollout.py --logdir ./data/fetchpush-ppomk-3 --num_rollouts 5


python hyperparameter_sweep.py --script_name ppo/ppo_koopman.py --params_path params/ppo_params_hypsweep.json