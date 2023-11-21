#!/bin/sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/11.8/targets/x86_64-linux/lib/
module load cuda-11.8
module load gcc-7.4

python main.py  --task='find_prompts' --exp_name='mt_t5_1'
python main.py --task slicing --exp_name='mt_t5_1'
python main.py --task prompt_analysis --exp_name='mt_t5_1'
python main.py --task label --exp_name='mt_t5_1'