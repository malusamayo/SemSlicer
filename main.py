from utils.parseArgument import parseArg

def main():
    args = parseArg()
    if args.task == "run_model":
        from runModel import run_model
        run_model(args)
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