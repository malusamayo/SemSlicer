from .utils.parseArgument import parseArg
from .utils.config import config
from .utils.log import get_logger
from .utils.file import read_txt_file, read_csv_file
from .slicer import Slicer
from .promptgen.prompt import PromptGenerator
import os

logger = get_logger("INFO", "main")

def main():
    args = parseArg()
    result_path = os.path.join("result", args.exp_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    config.update_path(args.exp_name)

    logger.info("Start running task: {exp}".format(exp=args.exp_name))

    keywords = read_txt_file(config["EXPERIMENT"]["KEYWORDS_PATH"])
    data = read_csv_file(config["EXPERIMENT"]["DATA_PATH"])
    if args.task == "run_model":
        from .inference import run_model
        run_model()
    elif args.task == "find_prompts":
        promptGen = PromptGenerator()
        promptGen.find_prompts_list(keywords)
    elif args.task == "slicing":
        slicer = Slicer()
        test_data = data.sample(n=config["SLICING"]["SAMPLE_SIZE"], random_state=42)
        if config["SLICING"]["FEW_SHOT"]:
            slicer.generate_few_shot_example_batch(test_data, keywords, method="usp")
        slicer.annotate_batch(test_data, keywords, prompt_existed=False, add_few_shot=config["SLICING"]["FEW_SHOT"])
    elif args.task == "label":
        slicer = Slicer()
        slicer.annotate_batch(data, keywords, prompt_existed=True)


if __name__ == "__main__":
    main()