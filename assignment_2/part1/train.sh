#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 01:00:00
#SBATCH -o fibonacci_%A.output
#SBATCH -e fibonacci_%A.error

module load python/3.5.2
module load cuda/8.0.44
module load cudnn/8.0-v6.0

python3 train.py --print_every 50 --summary_path ./summaries/test50 --learning_rate 0.0025 --model_type LSTM --input_length 50

