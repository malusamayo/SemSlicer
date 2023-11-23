import yaml
import os

def read_yaml_config(path, args, encoding="utf8"):
    with open(path, mode='r', encoding=encoding) as f:
        config = yaml.safe_load(f)
    for stage, options in config.items():
        for name, value in options.items():
            if name in ["RESULT_PATH", "TMP_PATH", "OUTPUT_PATH", "CSV_PATH", "DATASET_PATH", "PROMPT_PATH"]:
                config[stage][name] = os.path.join(result_path, config[stage][name])
    return config

result_path = os.path.join("result", "testbed")
config = read_yaml_config("./config.yaml", args=None)