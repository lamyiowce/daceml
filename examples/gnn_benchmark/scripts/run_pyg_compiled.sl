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
export PYTHONPATH=/users/jbazinsk/daceml:$PYTHONPATH


do_test=

model=gat
datasets="cora ogbn-arxiv pubmed citeseer flickr reddit"
modes="dgl"
hidden_sizes="8 16 32 64 128"
input_sizes="" # 128 is computed anyway.


echo "Running model " $model
for dataset in $datasets; do
  echo "Running dataset " $dataset

  # No input grad.
  outfile=./$(date +%d.%m.%H.%M)-pyg-$model-$dataset-$SLURM_JOB_ID.csv
  common_args="--mode benchmark --data $dataset --outfile $outfile --model $model --backward"
  for hidden in $hidden_sizes; do
    echo "Hidden " $hidden
    for m in $modes; do
      $do_test python torch_v2.py $common_args --hidden $hidden --backward --torch $m
    done
  done
#  for input_size in $input_sizes; do
#    echo "Input size " $input_size
#    for m in $modes; do
#      $do_test python torch_v2.py $common_args --hidden 128 --force-num-features $input_size --torch $m
#    done
#  done

  # With input grad.
#  outfile=./$(date +%d.%m.%H.%M)-pyg-$model-$dataset-input-grad-$SLURM_JOB_ID.csv
#  common_args="--mode benchmark --data $dataset --outfile $outfile --model $model --backward --input-grad"
#  for hidden in $hidden_sizes; do
#    echo "Hidden " $hidden
#    for m in $modes; do
#      $do_test python torch_v2.py $common_args --hidden $hidden --backward --torch $m
#    done
#  done
#  for input_size in $input_sizes; do
#    echo "Input size " $input_size
#    for m in $modes; do
#      $do_test python torch_v2.py $common_args --hidden 128 --force-num-features $input_size --torch $m
#    done
#  done
done

echo "Done :)"
