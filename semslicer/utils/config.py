import yaml
import os

def read_yaml_config(path, encoding="utf8"):
    with open(path, mode='r', encoding=encoding) as f:
        config = yaml.safe_load(f)
    return config


class Config:
    def __init__(self):
        self.config = None

    def read_config(self, path):
        self.config = read_yaml_config(path)

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value

    def update_path(self, exp_name):
        result_path = os.path.join("result", exp_name)
        for stage, options in self.config.items():
            for name, value in options.items():
                if name in ["PROMPT_PATH", "SLICE_RESULT_PATH", "FINAL_PROMPT_PATH", "FINAL_RESULT_PATH", "FEW_SHOT_PATH"]:
                    self.config[stage][name] = os.path.join(result_path, self.config[stage][name])
    
    def __str__(self):
        return str(self.config)

config = Config()