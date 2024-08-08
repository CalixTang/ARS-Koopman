# Koopman-RL

This is a repository containing code for the Koopman-RL project in the [STAR Lab](https://star-lab.cc.gatech.edu/). This repo was originallly forked from the [CIMER](https://github.com/GT-STAR-Lab/CIMER) project's repo. 



## Environment Setup


1. Clone this repository with submodules and cd into the project directory.
```
git clone --recursive https://github.com/CalixTang/Koopman-RL.git
cd Koopman-RL
```

2. Set up and activate the conda environment.
```
conda env create -f env.yml
conda activate koopmanrl-env
```

3. Set up project packages
```
cd utils
pip install -e .
cd ../ARS
pip install -e .
cd ..
```

## Training and Visualization

### ARS

To train a policy with ARS algorithm, use the ars.py script in the ARS folder. Example usages are below.

Default parameters (not recommended, but should run fine):
```
python ARS/ARS/ars.py
```

You can provide parameters manually in accordance with defined parameters in ars.py's `__main__` section (also not recommended to use this alone):
```
python ARS/ARS/ars.py --task_id Hopper-v4 --policy_type linear --run_name example_run
```

Finally, you can pass in a list of parameters from a json file (recommended). We also recommend changing the run name manually if using this approach:
```
python ARS/ARS/ars.py --params_path params/params.json
```

When using `--params_path`, the only CLI-assigned parameter that will overwrite what's in the params json is the `run_name` argument. 

### PPO
Training a policy with the PPO algorithm is very similar to training with ARS.

Default parameters (not recommended):
```
python PPO/ppo_koopman.py
```

With manually-provided parameters(not recommended):
```
python PPO/ppo_koopman.py --task_id Hopper-v4 --total_timesteps 1e6 --num_envs 4 --policy_type minkoopman --run_name example_run
```

With a json file (recommended)
```
python ARS/ARS/ars.py --params_path params/ppo_params.json
```

### Running with SLURM
You can run training jobs (and other compute jobs) on HPC clusters via SLURM. This repo has a provided SLURM script in `run_job.sbatch` which is currently configured for use on GT's PACE Phoenix cluster. For a guide to SLURM on PACE Phoenix, click [here](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0041998).

```
sbatch run_job.sbatch
```

### Hyperparameter sweep
An example hyperparameter sweep python script is provided. To use it, run
```
python hyperparameter_sweep.py --script_name ppo/ppo_koopman.py --params_path params/ppo_params_hypsweep.json
```
(Note: use this carefully)

### Visualization
You can record rollouts of trained policies with `ARS/ARS/record_rollout.py` and `PPO/ppo_record_rollout.py`. You will need to at least provide the output directory where the policy's weight file lives. 

For ARS-trained policies:
```
python ARS/ARS/record_rollout.py --logdir data/example_run --num_rollouts 20
```

For PPO-trained policies:
```
python PPO/ppo_record_rollout.py --logdir data/example_run --num_rollouts 20
```

### Note
You may find it useful to create an output directory (e.g. `data`) and a logging directory (e.g. `out`) right under the project root folder. The `--run_name` argument is used to create a subdirectory under the `--dir_path` location (which is `data` by default) to hold the output files of a training run. The training scripts currently print current progress to stdout, so you may find it useful to redirect this to a log file (e.g. `python ARS/ARS/ars.py > out/example_run.out`).
