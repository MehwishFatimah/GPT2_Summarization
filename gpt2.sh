#!/bin/bash
#SBATCH --job-name=gpt2-en
#SBATCH --output=/hits/basement/nlp/fatimamh/gpt2b_out/err-out/out-%j 
#SBATCH --error=/hits/basement/nlp/fatimamh/gpt2b_out/err-out/err-%j
#SBATCH --time=14-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=pascal-deep.p

module load CUDA/10.1.243-GCC-8.3.0

. /home/fatimamh/anaconda3/etc/profile.d/conda.sh
conda activate hugging_face
python /hits/basement/nlp/fatimamh/hf_summarization_v2/main.py 

