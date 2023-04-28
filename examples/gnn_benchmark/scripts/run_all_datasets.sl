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

do_test=echo

export DACE_compiler_cuda_max_concurrent_streams=-1
outfile=./out-$(hostname -s)-$(date +%d.%m.%H.%M)-$SLURM_JOB_ID.csv
model=gcn
flags="--persistent-mem --opt"
for dataset in cora ogbn-arxiv flickr reddit pubmed; do
  echo "Running dataset " $dataset
  for format in csr coo csr_reorder; do
    echo "Running format " $format
    for hidden in 8 32 128 512; do
      echo "Hidden " $hidden
      rm -rf .dacecache
      $do_test python benchmark.py --data $dataset --hidden $hidden --outfile $outfile --model $model $flags
    done
  done
done

