#! /bin/bash

#SBATCH --job-name=gcn
#SBATCH --nodes=23
#SBATCH --ntasks-per-node=48
#SBATCH --time=168:00:00
#SBATCH -o gcn_mpi.log

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=chaoyang.he@usc.edu

srun -n 972 /mnt/shared/etc/anaconda3/bin/python3 /mnt/shared/home/bipartite-graph-learning/automl.py