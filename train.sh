#!/usr/bin/env bash

declare MODEL=${1-'mtl'} # cnn / vae / mtl
declare EPOCH=${2-'200'}
declare SEED=${3-'123456'}

python train.py --model_name=${MODEL} --epoch=${EPOCH} --seed=${SEED}
