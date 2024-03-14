import argparse

def parseArg():
    parser = argparse.ArgumentParser()
    addArg(parser=parser)
    args=parser.parse_args()
    return args

def addArg(parser):
    parser.add_argument("--task", choices=["slicing", "find_prompts", "run_model", "prompt_analysis", "label"], required=True,
                        help="Selected from: [slicing, find_prompts, run_model, prompt_analysis, label]")
    parser.add_argument("--verbose", choices=["DEBUG", "INFO", "WARN"], default="INFO", \
                        help="Selected from: [debug, info, warn]")
    parser.add_argument("--exp_name", type=str, default="exp", help="experiment name")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="config file path")
    parser.add_argument("--data_path", type=str, default="data.csv", help="data file path")
    parser.add_argument("--keyword_path", type=str, default="keywords.csv", help="keyword file path")