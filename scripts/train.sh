echo Activating environment
source /research/milsrg1/user_workspace/efb48/miniconda3/bin/activate
conda activate dit
echo Python version used:
python -V
echo Starting evaluation...
python ./train.py
echo ...evaluation finished
