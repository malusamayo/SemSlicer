#!/bin/sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/11.8/targets/x86_64-linux/lib/
module load cuda-11.8
module load gcc-7.4

# python main.py --task slicing
gcc -c -o src/Binary1dSignalModel.o src/Binary1dSignalModel.cpp -fPIC
gcc -c -o src/BinaryModel.o src/BinaryModel.cpp -fPIC
gcc -c -o src/BinaryNdSignalModel.o src/BinaryNdSignalModel.cpp -fPIC
gcc -c -o src/BinarySignalModel.o src/BinarySignalModel.cpp -fPIC
gcc -c -o src/Model.o src/Model.cpp -fPIC
gcc -c -o src/annmodel.o src/annmodel.cpp -fPIC
gcc -c -o src/utils.o src/utils.cpp -fPIC
gcc -shared -o cubamcpp.so src/Binary1dSignalModel.o src/BinaryModel.o src/BinaryNdSignalModel.o src/BinarySignalModel.o src/Model.o src/annmodel.o src/utils.o -fPIC

python main.py --task prompt_analysis
python main.py --task label