#!/bin/bash
#SBATCH --job-name=gnn_pyg_compiled      # Job name
#SBATCH --time=01:00:00              # Wall time limit (days-hrs:min:sec)
#SBATCH --output=gnn_pyg_compiled_%j.log     # Path to the standard output and error files relative to the working directory
#SBATCH -p intelv100
#SBATCH --account=g34
#SBATCH --gpus=1

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""

source /users/jbazinsk/miniconda3/bin/activate
conda activate torch2
module load cuda/11.8.0
export LIBRARY_PATH=/users/jbazinsk/miniconda3/envs/torch2/lib/:$LIBRARY_PATH

rm -rf ./.dacecache

do_test=

model=gat
datasets="cora ogbn-arxiv"
modes="edge_list edge_list_compiled dgnn dgnn_compiled csr"
hidden_sizes="8 16 32 64 128 256"

echo "Running model " $model
for dataset in $datasets; do
  echo "Running dataset " $dataset
  outfile=./$(date +%d.%m.%H.%M)-pyg-$model-$dataset-$SLURM_JOB_ID.csv
  for hidden in $hidden_sizes; do
    echo "Hidden " $hidden
    rm -rf .dacecache
    for m in $modes; do
      $do_test python torch_v2.py --mode benchmark --data $dataset --hidden $hidden --outfile $outfile --model $model --backward --torch $m
    done
  done
done

echo "Done :)"
