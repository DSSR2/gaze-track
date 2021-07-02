#!/bin/bash
#SBATCH --time=1-00:00        
#SBATCH --nodes=1             
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --job-name=add_kp
#SBATCH --output=add_kp.out
#SBATCH --mem=90G

module load python/3.8
module load httpproxy

cd $SLURM_TMPDIR
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install dlib
module load scipy-stack
module load opencv


pip install torch --no-index
pip install torchvision --no-index
pip install comet-ml --no-index
pip install --upgrade setuptools
pip install scikit-image
pip install imutils 

tar -xf ~/projects/def-skrishna/dssr/gazetrack.tar.gz -C .
echo untar done
scp ~/projects/def-skrishna/gaze-track/Utils/add_eye_kp.py .
git clone https://github.com/italojs/facial-landmarks-recognition.git
scp facial-landmarks-recognition/shape_predictor_68_face_landmarks.dat .

ls

echo starting py
python add_eye_kp.py --dir ./gazetrack/ --workers 20 --p ./shape_predictor_68_face_landmarks.dat

tar -c --use-compress-program=pigz -f gazetrack.tar.gz ./gazetrack/
scp  gazetrack.tar.gz ~/projects/def-skrishna/dssr
