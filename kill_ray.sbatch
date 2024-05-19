#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 8
#SBATCH --mem-per-cpu=16G 
#SBATCH -A gts-hravichandar3-starlab
#SBATCH -J kill_ray_slurm
#SBATCH -t 3:00:00 
#SBATCH -o log.out

#run the application:
#module load mesa
#module load anaconda3 

CAUSAL_COMM_FOLDER=~/scratch/CIMER/

cd $CAUSAL_COMM_FOLDER
ray start --head
#export MKL_NUM_THREADS=1
srun python ./ARS/code/kill_ray.py

ray stop
