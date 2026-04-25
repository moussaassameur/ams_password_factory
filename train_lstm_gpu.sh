#!/bin/bash
#SBATCH --job-name=lstm_train
#SBATCH --output=logs/lstm_%j.out
#SBATCH --error=logs/lstm_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

cd ~/ams_password_factory/src

python3 -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')"

python3 train_lstm.py
