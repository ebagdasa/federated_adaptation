#!/bin/bash
#SBATCH -J med_
#SBATCH -o med_.o%j
#SBATCH -e med_.o%j
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=10000
#SBATCH -t 12:00:00
#SBATCH --partition=desa  --gres=gpu:1


# OPTIONAL: uncomment this if you need to copy a dataset over to scratch
#    This checks to see if the dataset already exists
# if [ ! -d /scratch/datasets/shard_by_author.zip ]; then
#    cp  /home/ty367/federated/data/shard_by_author.zip /scratch/datasets/shard_by_author.zip
#    cd /scratch/datasets/
#    unzip shard_by_author.zip
# fi

# cp  /home/ty367/federated/data/corpus_80000.pt.tar /scratch/datasets/corpus_80000.pt.tar
# if [ ! -d /scratch/datasets/corpus_80000.pt.tar ]; then
#    cp  /home/ty367/federated/data/corpus_80000.pt.tar /scratch/datasets/corpus_80000.pt.tar
# fi

# CUDA_VISIBLE_DEVICES=2 nohup python -u training1.py --name image >log_eval_av_0.01.txt 2>&1 &
# python /home/ty367/federated/training1.py --name text --params /home/ty367/federated/utils/words1.yaml
python training1.py --name image
