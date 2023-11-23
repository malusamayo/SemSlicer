from .utils.parseArgument import parseArg
from .slicer import Slicer
from .promptgen import PromptGenerator
import os

def main():
    args = parseArg()
    result_path = os.path.join("result", args.exp_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    args.result_path = result_path


    keywords = read_txt_file(config["EXPERIMENT"]["KEYWORDS_PATH"])
    data = read_csv_file(config["SLICING"]["DATASET_PATH"])
    if args.task == "run_model":
        return
    elif args.task == "find_prompts":
        promptGen = PromptGenerator()
        promptGen.find_prompts_list(keywords)
    elif args.task == "slicing":
        slicer = Slicer()
        test_data = df.sample(n=config["SLICING"]["SAMPLE_SIZE"], random_state=42)
        slicer.annotate_batch(test_data, keywords, prompt_existed=False)
    elif args.task == "label":
        slicer = Slicer()
        slicer.annotate_batch(df, keywords, prompt_existed=True)


if __name__ == "__main__":
    main()