#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G
#SBATCH --time=0:15:0
python dataprocessing.py
