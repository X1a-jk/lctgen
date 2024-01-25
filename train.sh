#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3,5,6
export TORCH_DISTRIBUTED_DEBUG=INFO
python lctgen/main.py  --run-type train --exp-config cfgs/train.yaml

