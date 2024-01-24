#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7,9,11
export TORCH_DISTRIBUTED_DEBUG=INFO
python lctgen/main.py  --run-type eval --exp-config cfgs/inference.yaml
