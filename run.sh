#!/bin/sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/11.8/targets/x86_64-linux/lib/
module load cuda-11.8
module load gcc-7.4

python -m semslicer.main  --task='find_prompts' --exp_name='testbed_t5_1'
python -m semslicer.main --task slicing --exp_name='testbed_t5_1'
python -m semslicer.main --task label --exp_name='testbed_t5_1'