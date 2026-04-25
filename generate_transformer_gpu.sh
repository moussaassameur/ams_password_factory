#!/bin/bash
#SBATCH --job-name=trf_gen
#SBATCH --output=logs/trf_gen_%j.out
#SBATCH --error=logs/trf_gen_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

echo "=== Démarrage génération : $(date) ==="
echo "Nœud : $SLURMD_NODENAME"

cd ~/ams_password_factory/src

python3 -c "
import torch
print('PyTorch :', torch.__version__)
print('CUDA    :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU     :', torch.cuda.get_device_name(0))
"

echo ""
echo "=== Génération 10k / 100k / 1M ==="
python3 generate_transformer_decoder.py

echo ""
echo "=== Évaluation Phase 3 ==="
python3 evaluate_phase3.py

echo ""
echo "=== Terminé : $(date) ==="
