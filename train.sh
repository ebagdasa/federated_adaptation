#!/bin/bash
#SBATCH -J adapt_med
#SBATCH -o adapt_med.o%j
#SBATCH -e adapt_med.o%j
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=10000
#SBATCH -t 480:00:00
#SBATCH --partition=default_gpu  --gres=gpu:1


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

# CUDA_VISIBLE_DEVICES=1 nohup python -u training1.py --name image >log_av_sc0.01.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python /home/ty367/federated/training1.py --name text --params /home/ty367/federated/utils/words1.yaml >text_av_fbr.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python /home/ty367/federated/training1.py --name text --params /home/ty367/federated/utils/words1.yaml
# python adapt.py --name image --params /home/ty367/federated/utils/adapt_image.yaml
python /home/ty367/federated/adapt.py --name text --params /home/ty367/federated/utils/adapt_text.yaml
