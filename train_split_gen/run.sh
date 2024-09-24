#!/bin/bash

#SBATCH --job-name=iemocap
#SBATCH --qos=a100-4
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/dilab/Seong/peft-ser/train_split_gen/output.out

# python3 iemocap6_audio.py
python3 iemocap6.py
