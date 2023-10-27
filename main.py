from utils.parseArgument import parseArg
import os

def main():
    args = parseArg()
    result_path = os.path.join("result", args.exp_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    args.result_path = result_path
    if args.task == "run_model":
        from runModel import run_model
        run_model(args)
    elif args.task == "find_prompts":
        from prompt_finder import find_prompts
        find_prompts(args)
    elif args.task == "slicing":
        from slicing import slicing
        slicing(args)
    elif args.task == "prompt_analysis":
        from prompt_analysis import prompt_analysis
        prompt_analysis(args)
    elif args.task == "label":
        from label import label
        label(args)


if __name__ == "__main__":
    main()