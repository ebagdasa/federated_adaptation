#!/bin/bash
#SBATCH -J adapt
#SBATCH -o adapt.o%j
#SBATCH -e adapt.o%j
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=10000
#SBATCH -t 20:00:00
#SBATCH --partition=default_gpu  --gres=gpu:1

# CUDA_VISIBLE_DEVICES=0 nohup python training.py --name text --params ./utils/words.yaml >train_debug.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python adapt.py --name text --params ./utils/adapt_text.yaml >train_debug.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python training.py --name image --params ./utils/params.yaml >train_debug.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python adapt.py --name image --params ./utils/adapt_image.yaml >train_debug.txt 2>&1 &
# python adapt.py --name image --params /home/ty367/federated/utils/adapt_image.yaml
python /home/ty367/federated/adapt.py --name text --params /home/ty367/federated/utils/adapt_text.yaml

