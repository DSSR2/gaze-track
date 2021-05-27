#!/bin/bash
#SBATCH --time=1-00:00        
#SBATCH --nodes=1             
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=47000
#SBATCH --job-name=add_kp
#SBATCH --output=add_kp.out

#SBATCH --mail-user=dineshsathiaraj2@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


module load python/3.6
module load scipy-stack
module load httpproxy

cd $SLURM_TMPDIR
mkdir work
cd work
git clone https://github.com/DSSR2/gaze-track.git
mkdir mit-dataset
tar -xf ~/projects/dssr/gazecapture.tar -C mit-dataset
cd mit-dataset
cat *.tar.gz | tar zxf - -i
rm -rf *.tar.gz

cd ..
mkdir dataset
cd dataset
mkdir train test val

cd train
mkdir images meta

cd ../test
mkdir images meta

cd ../val
mkdir images meta

cd ../../gaze-track/Utils
python dataset_converter.py --threads 10 --dir ../../mit-dataset/ --out_dir ../../dataset/

tar -c --use-compress-program=pigz -f updated_dataset.tar.gz ../../dataset/
scp  updated_dataset.tar.gz ~/projects/def-skrishna/dssr

cd ../../dataset/train/images
echo train
ls | wc -l
cd ../../test/images
echo test
ls | wc -l
echo val
cd ../../val/images
ls | wc -l
