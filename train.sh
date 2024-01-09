#!/bin/bash
export CUDA_VISIBLE_DEVICES=5,6,7
python lctgen/main.py  --run-type train --exp-config cfgs/train.yaml
