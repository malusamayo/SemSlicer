#!/bin/sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/11.8/targets/x86_64-linux/lib/
module load cuda-11.8
module load gcc-7.4

python main.py --task slicing
python main.py --task prompt_analysis
python main.py --task label