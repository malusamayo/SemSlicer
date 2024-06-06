#!/bin/bash
#SBATCH --job-name=prompt-slicer
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=48GB

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/11.8/targets/x86_64-linux/lib/
module load cuda-11.8
module load gcc-7.4

declare -A data_paths
data_paths["cc"]="data/data/civil_comments_sampled.csv"
data_paths["amazon"]="data/data/amazon_reviews_sample.csv"
data_paths["hotel"]="data/data/hotel.csv"
data_paths["hc"]="data/data/hatecheck.csv"
# data_paths["sarcasm"]="data/data/sarcasm_news.csv"
# data_paths["vul"]="data/data/so_vul_sampled.csv"
# data_paths["mt-jaen"]="data/data/mt-jaen.csv"
# data_paths["accept"]="data/data/acceptability.csv"

declare -A keyword_paths
keyword_paths["cc"]="data/keywords/cc-keywords_v2.csv"
keyword_paths["amazon"]="data/keywords/amazon-keywords_v3.csv"
keyword_paths["hotel"]="data/keywords/hotel-keywords.csv"
keyword_paths["hc"]="data/keywords/hatecheck-keywords.csv"
# keyword_paths["sarcasm"]="data/keywords/sarcasm-keywords.csv"
# keyword_paths["vul"]="data/keywords/vul-keywords.csv"
# keyword_paths["mt-jaen"]="data/keywords/mt-keywords.csv"
# keyword_paths["accept"]="data/keywords/acc-keywords.csv"

dataset=$1
post_fix=''
for method in 'zs' 'fs', 'fs-div', 'fs-div-teach', 'fs-syn' 'zs-gen' 'zs-ref'; do
    exp_name="${dataset}_fn_${method}${post_fix}"
    config="data/config/config_${method}.yaml"

    python -m semslicer.main  --task='find_prompts' --exp_name="${exp_name}"\
        --config="${config}"\
        --data_path="${data_paths[$dataset]}"\
        --keyword_path="${keyword_paths[$dataset]}"

    python -m semslicer.main  --task='slicing' --exp_name="${exp_name}"\
        --config="${config}"\
        --data_path="${data_paths[$dataset]}"\
        --keyword_path="${keyword_paths[$dataset]}"
done



