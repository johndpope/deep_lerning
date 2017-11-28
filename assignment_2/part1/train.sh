#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 01:00:00
#SBATCH -o fibonacci_%A.output
#SBATCH -e fibonacci_%A.error

module load python/3.5.2
module load cuda/8.0.44
module load cudnn/8.0-v6.0

python3 train.py --print_every 50 --summary_path ./summaries/test20 --learning_rate 0.0025 --input_length 20
python3 train.py --print_every 50 --summary_path ./summaries/test25 --learning_rate 0.0025 --input_length 25
python3 train.py --print_every 50 --summary_path ./summaries/test30 --learning_rate 0.0025 --input_length 30
python3 train.py --print_every 50 --summary_path ./summaries/test35 --learning_rate 0.0025 --input_length 35
python3 train.py --print_every 50 --summary_path ./summaries/test40 --learning_rate 0.0025 --input_length 40
python3 train.py --print_every 50 --summary_path ./summaries/test45 --learning_rate 0.0025 --input_length 45
python3 train.py --print_every 50 --summary_path ./summaries/test50 --learning_rate 0.0025 --input_length 50
