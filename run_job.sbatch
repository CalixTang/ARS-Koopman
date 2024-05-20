#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH --mem-per-cpu=64G 
#SBATCH -A gts-hravichandar3-starlab
#SBATCH -J ARS_slurm
#SBATCH -t 12:00:00 
#SBATCH -o reports/report-%j.out 

#run the application:
module load mesa
#module load anaconda3 


cd $SLURM_SUBMIT_DIR
#ray stop
ray start --head
#export MKL_NUM_THREADS=1
srun python ./ARS/code/ars_koopman.py
ray stop
