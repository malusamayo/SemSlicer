from utils.parseArgument import parseArg
from slicing import slicing
from dataprocess import dataprocess
from runModel import run_model
from prompt_analysis import prompt_analysis
from label import label
def main():
    args = parseArg()
    if args.task == "slicing":
        slicing(args)
    elif args.task == "dataprocess":
        dataprocess(args)
    elif args.task == "run_model":
        run_model(args)
    elif args.task == "prompt_analysis":
        prompt_analysis(args)
    elif args.task == "label":
        label(args)


if __name__ == "__main__":
    main()