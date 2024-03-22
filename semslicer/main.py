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
    config.load_data_and_keyword_path(args.data_path, args.keyword_path)
    config.update_path(args.exp_name)

    logger.info("Start running task: {exp}".format(exp=args.exp_name))
    logger.info("Config:\n{config}".format(config=config))

    keyword_df = read_csv_file(config["EXPERIMENT"]["KEYWORDS_PATH"])
    keywords = keyword_df["keyword"].tolist()
    data = read_csv_file(config["EXPERIMENT"]["DATA_PATH"])

    logger.info("Keywords: {keywords}".format(keywords=keywords))

    if args.task == "find_prompts":
        promptGen = PromptGenerator(instruction_source=config["INSTRUCTION"]["SOURCE"])
        promptGen.find_prompts_list(keyword_df)
        slicer = Slicer(model_name="dummy")
        if config["EXAMPLES"]["USE_FEW_SHOT"]:
            if not os.path.exists(config["EXPERIMENT"]["FEW_SHOT_PATH"]):
                slicer.generate_few_shot_example_batch(data, keywords, 
                    num=config["EXAMPLES"]["FEW_SHOT_SIZE"], 
                    input_sampling_strategy=config["EXAMPLES"]["SAMPLE_STRATEGY"],
                    output_label_source=config["EXAMPLES"]["LABEL_SOURCE"], 
                    synthesize=config["EXAMPLES"]["SYNTHESIZE"])
    elif args.task == "slicing":
        slicer = Slicer(model_name=config["SLICING"]["MODEL_NAME"])
        if config["SLICING"]["SAMPLING"]:
            data = data.sample(n=config["SLICING"]["SAMPLE_SIZE"], random_state=42)
        slicer.annotate_batch(data, keywords, 
            use_calibrate=config["SLICING"]["CALIBRATE"], 
            add_few_shot=config["EXAMPLES"]["USE_FEW_SHOT"],)
            # use_cache=True)


if __name__ == "__main__":
    main()