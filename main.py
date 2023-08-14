from utils.parseArgument import parseArg
from slicing import slicing
def main():
    args = parseArg()
    if args.task == "slicing":
        slicing(args)

if __name__ == "__main__":
    main()