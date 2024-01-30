#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,7
export TORCH_DISTRIBUTED_DEBUG=INFO
python lctgen/main.py  --run-type eval --exp-config cfgs/inference.yaml

