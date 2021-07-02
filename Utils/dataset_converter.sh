#!/bin/bash
#SBATCH --time=1-00:00        
#SBATCH --nodes=1             
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=47000
#SBATCH --job-name=ds_conv
#SBATCH --output=ds_conv.out

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
tar -xf ~/projects/def-skrishna/dssr/gazecapture.tar -C mit-dataset
cd mit-dataset
cat *.tar.gz | tar zxf - -i
rm -rf *.tar.gz

cd ..
mkdir dataset

cd dataset
mkdir train test val

mkdir train/images train/meta
mkdir test/images test/meta
mkdir val/images val/meta

cd ../gaze-track/Utils
python dataset_converter_google.py --threads 10 --dir ../../mit-dataset/ --out_dir ../../dataset/

tar -c --use-compress-program=pigz -f updated_dataset.tar.gz ../../dataset/
scp  updated_dataset.tar.gz ~/projects/def-skrishna/dssr

cd ../../dataset/train/images
echo train images: 
ls | wc -l
cd ../../test/images
echo test images:
ls | wc -l
echo val images: 
cd ../../val/images
ls | wc -l
