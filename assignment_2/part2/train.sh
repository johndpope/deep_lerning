#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 02:30:00
#SBATCH -o fibonacci_%A.output
#SBATCH -e fibonacci_%A.error

module load python/3.5.2
module load cuda/8.0.44
module load cudnn/8.0-v6.0

python3 train.py --txt_file books/book_EN_democracy_in_the_US.txt --print_every 200
