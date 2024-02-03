#!/bin/bash
export CUDA_VISIBLE_DEVICES=8,11,10,7
export TORCH_DISTRIBUTED_DEBUG=INFO
python lctgen/main.py  --run-type train --exp-config cfgs/train.yaml

