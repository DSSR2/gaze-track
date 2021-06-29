#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:2
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=90G
#SBATCH --output=lit_train.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_BLOCKING_WAIT=1

module load python/3.8
module load httpproxy
pwd
cd $SLURM_TMPDIR
virtualenv ./env
source ./env/bin/activate

module load opencv
module load scipy-stack

pip install pillow pandas tensorboard torch torchvision comet-ml pytorch-lightning --no-index
pip install --upgrade setuptools
pip install scikit-image imutils tqdm

tar -xf  ~/projects/dataset_ip6s_kps.tar.gz -C .
echo untar done
ls 
scp -r  ~/projects/gaze-track/Model/Lightning .
echo copy done

ls
cd Lightning

python lit_train.py --dataset_dir ../datasetip6s/ --save_dir ~/projects/dssr/trained_models/lit/ --gpus 2 --epochs 100