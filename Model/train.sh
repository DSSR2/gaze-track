#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --output=train.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python/3.8
module load httpproxy
pwd
cd $SLURM_TMPDIR
virtualenv --no-download ./env
source ./env/bin/activate

module load opencv
module load scipy-stack

pip install pillow pandas tensorboard torch torchvision comet-ml --no-index
pip install --upgrade setuptools
pip install scikit-image imutils tqdm

echo dependencies installed

git clone https://github.com/DSSR2/gaze-track.git
cd gaze-track

scp -r  ~/projects/dssr/dataset_kps.tar.gz .
tar -xf dataset_kps.tar.gz

echo untar done


cd Model
python gazetrack_train.py --dataset_dir ../dataset/ --save_dir ./trained_models/ --csv_dir ./trained_models/ --model_name gazetrack --gpu 1 --workers 10 --epochs 200

