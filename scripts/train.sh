echo Activating environment
source /research/milsrg1/user_workspace/efb48/miniconda3/bin/activate
conda activate dit
echo Python version used:
python -V
echo Starting evaluation...
python ./evaluate_mp.py /data/milsrg1/huggingface/cache/efb48/diffsep/checkpoint.pt --split libri-clean
echo ...evaluation finished
