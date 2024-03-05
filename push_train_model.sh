#!/bin/bash

#SBATCH --job-name=Job_python
#SBATCH --output=resultat.txt
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --mem=4096
#SBATCH --mail-type=ALL
#SBATCH --mail-user = chiraze.drine@polymtl.ca
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1

# Charger l'environnement module si nécessaire
source /etc/profile.d/modules.sh

# Charger le module Python/Anaconda
module load anaconda3
conda activate gdcvae

# Exécuter le script Python
python train_model.py