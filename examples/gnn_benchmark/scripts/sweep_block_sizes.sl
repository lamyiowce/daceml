#!/bin/bash
#SBATCH --job-name=gnn_block_sizes      # Job name
#SBATCH --time=02:00:00              # Wall time limit (days-hrs:min:sec)
#SBATCH --output=gnn_block_sizes_%j.log     # Path to the standard output and error files relative to the working directory
#SBATCH -w ault24
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

do_test=

export DACE_default_build_folder=./.dacecache-$SLURM_JOB_ID
export DACE_compiler_cuda_max_concurrent_streams=-1
model=gcn
formats="csr_adapt coo_adapt"
backward=--backward
datasets="cora ogbn-arxiv"

block_sizes="512,1,1 32,1,1 64,8,1"
hidden_sizes="32 128 512"
echo "Running model " $model
for dataset in $datasets; do
  echo "Running dataset " $dataset
  outfile=./$(date +%d.%m.%H.%M)-block-sizes-$model-$dataset-$SLURM_JOB_ID.csv
  for block in $block_sizes; do
    echo "Block " $block
    export DACE_compiler_cuda_default_block_size=$block
    for hidden in $hidden_sizes; do
      echo "Hidden " $hidden
      rm -rf $DACE_default_build_folder
      for format in $formats; do
        $do_test python main.py --tag ${block//,/_} --mode benchmark --data $dataset --hidden $hidden --outfile $outfile --model $model --impl $format --torch csr $backward
      done
    done
  done
done

echo "Done :)"
