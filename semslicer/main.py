from .utils.parseArgument import parseArg
from .utils.config import config
from .utils.log import get_logger
from .utils.file import read_txt_file, read_csv_file
from .slicer import Slicer
from .promptgen.generator import PromptGenerator
import os

logger = get_logger("INFO", "main")

def main():
    args = parseArg()
    result_path = os.path.join("result", args.exp_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    config.read_config(args.config_path)
    config.update_path(args.exp_name)

    logger.info("Start running task: {exp}".format(exp=args.exp_name))
    logger.info("Config:\n{config}".format(config=config))

    keyword_df = read_csv_file(config["EXPERIMENT"]["KEYWORDS_PATH"])
    keywords = keyword_df["keyword"].tolist()
    data = read_csv_file(config["EXPERIMENT"]["DATA_PATH"])

    if args.task == "find_prompts":
        promptGen = PromptGenerator()
        promptGen.find_prompts_list(keyword_df)
        slicer = Slicer(model_name="dummy")
        if config["SLICING"]["FEW_SHOT"]:
            if not os.path.exists(config["EXPERIMENT"]["FEW_SHOT_PATH"]):
                slicer.generate_few_shot_example_batch(data, keywords, method="random")
    elif args.task == "slicing":
        # slicer = Slicer(model_name="llama2", model_size="13b-chat")
        slicer = Slicer(model_name="flan-t5", model_size="xxl")
        if config["SLICING"]["SAMPLING"]:
            data = data.sample(n=config["SLICING"]["SAMPLE_SIZE"], random_state=42)
        slicer.annotate_batch(data, keywords, use_calibrate=config["SLICING"]["CALIBRATE"], add_few_shot=config["SLICING"]["FEW_SHOT"])


if __name__ == "__main__":
    main()