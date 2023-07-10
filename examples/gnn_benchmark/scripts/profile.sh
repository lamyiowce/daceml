#!/bin/bash

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

source /users/jbazinsk/miniconda3/bin/activate
conda activate env14
module load cuda/11.4.0


export LIBRARY_PATH=/users/jbazinsk/miniconda3/envs/env14/lib/:$LIBRARY_PATH

export PYTHONPATH=/users/jbazinsk/daceml:$PYTHONPATH

export ORT_RELEASE=/users/jbazinsk/onnxruntime-daceml-patched
export DACE_default_build_folder=./.dacecache-ncompute
export DACE_compiler_cuda_default_block_size=64,8,1
export DACE_compiler_cuda_max_concurrent_streams=-1
model=gat_single_layer
backward=--backward

format=coo
dataset=ogbn-arxiv
hidden=64
python main.py --mode benchmark --data $dataset --hidden $hidden --model $model --impl $format --torch none $backward

echo "Done :)"
