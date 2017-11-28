#!/bin/bash

python3 train.py --print_every 50 --summary_path ./summaries/test10 --learning_rate 0.0025 --input_length 10
python3 train.py --print_every 50 --summary_path ./summaries/test15 --learning_rate 0.0025 --input_length 15
python3 train.py --print_every 50 --summary_path ./summaries/test20 --learning_rate 0.0025 --input_length 20

python3 train.py --print_every 50 --summary_path ./summaries/test25 --learning_rate 0.0025 --input_length 25
python3 train.py --print_every 50 --summary_path ./summaries/test30 --learning_rate 0.0025 --input_length 30
python3 train.py --print_every 50 --summary_path ./summaries/test35 --learning_rate 0.0025 --input_length 35
python3 train.py --print_every 50 --summary_path ./summaries/test40 --learning_rate 0.0025 --input_length 40
python3 train.py --print_every 50 --summary_path ./summaries/test45 --learning_rate 0.0025 --input_length 45
python3 train.py --print_every 50 --summary_path ./summaries/test50 --learning_rate 0.0025 --input_length 50
