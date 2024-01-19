#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
export TORCH_DISTRIBUTED_DEBUG=INFO
python lctgen/main.py  --run-type train --exp-config cfgs/train.yaml
