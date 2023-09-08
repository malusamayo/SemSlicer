import argparse

def parseArg():
    parser = argparse.ArgumentParser()
    addArg(parser=parser)
    args=parser.parse_args()
    return args

def addArg(parser):
    parser.add_argument("--task", choices=["slicing", "dataprocess", "run_model", "prompt_analysis", "label"], required=True, \
                        help="Selected from: [slicing, dataprocess, run_model, prompt_analysis, label]")
    parser.add_argument("--verbose", choices=["DEBUG", "INFO", "WARN"], default="INFO", \
                        help="Selected from: [debug, info, warn]")