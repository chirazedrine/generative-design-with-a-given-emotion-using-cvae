#!/bin/bash

#SBATCH --job-name=MonJobPython
#SBATCH --output=resultat.txt
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=4096
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Charger l'environnement module si nécessaire
source /etc/profile.d/modules.sh

# Charger le module Python/Anaconda
module load anaconda3
conda activate gdcvae

# Exécuter le script Python
python train_model.py