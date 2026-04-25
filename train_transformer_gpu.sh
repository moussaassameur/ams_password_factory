#!/bin/bash
#SBATCH --job-name=trf_train
#SBATCH --output=logs/trf_train_%j.out
#SBATCH --error=logs/trf_train_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

echo "=== Démarrage : $(date) ==="
echo "Nœud : $SLURMD_NODENAME"

cd ~/ams_password_factory/src

# Vérification GPU
python3 -c "
import torch
print('PyTorch :', torch.__version__)
print('CUDA    :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU     :', torch.cuda.get_device_name(0))
    print('VRAM    :', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'Go')
"

echo ""
echo "=== Entraînement ==="
python3 train_transformer_decoder.py

echo ""
echo "=== Terminé : $(date) ==="
