#!/bin/bash
#SBATCH --job-name=gnn_benchmark      # Job name
#SBATCH --time=02:00:00              # Wall time limit (days-hrs:min:sec)
#SBATCH --output=single_layer_%j.log     # Path to the standard output and error files relative to the working directory
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
export DACE_compiler_cuda_default_block_size=64,8,1

model=gcn_single_layer
formats="coo_cached coo_adapt csr_adapt csc_adapt csr_coo_adapt-0.99 csc_cached"
datasets="cora ogbn-arxiv"
backward=--backward
hidden_sizes="8 16 32 64 128 256"

echo "Running model " $model
for dataset in $datasets; do
  echo "Running dataset " $dataset
  outfile=./$(date +%d.%m.%H.%M)-$model-$dataset-$SLURM_JOB_ID.csv
  for hidden in $hidden_sizes; do
    echo "Hidden " $hidden
    rm -rf $DACE_default_build_folder
    for format in $formats; do
      echo "Format " $format
      $do_test python main.py --mode benchmark --data $dataset --hidden $hidden --outfile $outfile --model $model --impl $format --torch none $backward
    done
  done
done

echo "Done :)"

