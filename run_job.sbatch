#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH --mem-per-cpu=48G 
#SBATCH -A gts-hravichandar3-starlab
#SBATCH -J ARS_slurm
#SBATCH -t 6:00:00 
#SBATCH -o reports/report-%j.out 

#run the application:
module load mesa
#module load anaconda3 

CAUSAL_COMM_FOLDER=~/scratch/CIMER/

cd $CAUSAL_COMM_FOLDER
#ray stop
ray start --head
#export MKL_NUM_THREADS=1
srun python ./ARS/code/ars_koopman.py
ray stop
