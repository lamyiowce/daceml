#!/bin/bash
#SBATCH --job-name=gnn_benchmark      # Job name
#SBATCH --time=01:00:00              # Wall time limit (days-hrs:min:sec)
#SBATCH --output=gnn_benchmark_%j.log     # Path to the standard output and error files relative to the working directory
#SBATCH -p intelv100
#SBATCH --account=g34
#SBATCH --gpus=1


echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""

source /users/jbazinsk/miniconda3/bin/activate
conda activate env14
module load cuda/11.4.0
export LIBRARY_PATH=/users/jbazinsk/miniconda3/envs/env/lib/:$LIBRARY_PATH

rm -rf ./.dacecache

do_test=

export DACE_compiler_cuda_max_concurrent_streams=-1
model=gcn
formats="csr coo csc"

datasets="cora ogbn-arxiv"

#hidden_sizes="8 32 128 512 2048"
hidden_sizes="8 16 32 64 128 256 512 1024"
echo "Running model " $model
for dataset in $datasets; do
  echo "Running dataset " $dataset
  outfile=./$(date +%d.%m.%H.%M)-$model-$dataset-$SLURM_JOB_ID.csv
  for hidden in $hidden_sizes; do
    echo "Hidden " $hidden
    rm -rf .dacecache
    $do_test python main.py --mode benchmark --data $dataset --hidden $hidden --outfile $outfile --model $model --impl none --backward
    $do_test python main.py --mode benchmark --data $dataset --hidden $hidden --outfile $outfile --model $model --impl $formats --torch none --backward
  done
done

