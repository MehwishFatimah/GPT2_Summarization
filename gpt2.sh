#!/bin/bash
#SBATCH --job-name=1v3-a16pgn
#SBATCH --output=/hits/basement/nlp/fatimamh/01_v3_cs_pgn_all/data/err-out/out-%j 
#SBATCH --error=/hits/basement/nlp/fatimamh/01_v3_cs_pgn_all/data/err-out/err-%j
#SBATCH --time=14-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=pascal-deep.p

module load CUDA/9.2.88-GCC-7.3.0-2.30

. /home/fatimamh/anaconda3/etc/profile.d/conda.sh
conda activate pytorch-gpu
python /hits/basement/nlp/fatimamh/01_v3_cs_pgn_all/pgn_m/main.py -ts -t -e 

