#!/bin/bash

#SBATCH --job-name=jupyter
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=5:00:00
#SBATCH --gres=gpu

module purge

singularity exec --nv \
	        --overlay /scratch/ab9738/fieldformer/overlay-25GB-500K.ext3:rw \
		/scratch/ab9738/fieldformer/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
		/bin/bash -c '
       		source /ext3/env.sh;
		port=8882;
		/usr/bin/ssh -N -f -R ${port}:localhost:${port} log-3;
		/usr/bin/ssh -N -f -R ${port}:localhost:${port} log-2;
		/usr/bin/ssh -N -f -R ${port}:localhost:${port} log-1;
		jupyter lab --no-browser --ip=0.0.0.0 --port ${port} --notebook-dir /scratch/ab9738 --NotebookApp.token="" --NotebookApp.password=""'

