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

# python -m semslicer.main  --task='slicing' --exp_name='sbic_v2_gpt3.5_0' --config='config_sbic.yaml'
# python -m semslicer.main  --task='slicing' --exp_name='hqa_t5_5' --config='config_hqa.yaml'
# python -m semslicer.main  --task='slicing' --exp_name='cc_t5_full_10' --config='config_cc.yaml'
# python -m semslicer.main  --task='slicing' --exp_name='mmlu_t5_1' --config='config_mmlu.yaml'
# python -m semslicer.main  --task='slicing' --exp_name='amazon_t5_full_6' --config='config_amazon.yaml'
# python -m semslicer.main  --task='find_prompts' --exp_name='yahoo_t5_0' --config='config_yahoo.yaml'
# python -m semslicer.main  --task='slicing' --exp_name='ge_t5_1' --config='config_emotion.yaml'
# python -m semslicer.main  --task='slicing' --exp_name='ge_t5_2' --config='config_emotion.yaml'
# python -m semslicer.main  --task='slicing' --exp_name='yahoo_t5_full_4' --config='config_yahoo.yaml'
# python -m semslicer.main  --task='slicing' --exp_name='irony_t5_0' --config='config_irony.yaml'
# python -m semslicer.main  --task='slicing' --exp_name='ait_t5_8' --config='config_ait.yaml'
# python -m semslicer.main  --task='slicing' --exp_name='sted_t5_0' --config='config_sted.yaml'

# python -m semslicer.main  --task='find_prompts' --exp_name='disamqa_llama_0'
# python -m semslicer.main --task slicing --exp_name='rte_t5_0'
# python -m semslicer.main --task slicing --exp_name='tweet-emo_t5_al_cali_0'
# python -m semslicer.main --task slicing --exp_name='disamqa_llama_al_30shot_0'
# python -m semslicer.main --task label --exp_name='disamqa_llama_0'/home/cyang3/slicing/result/




