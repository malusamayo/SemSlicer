import yaml

def read_yaml_config(path, encoding="utf8"):
    with open(path, mode='r', encoding=encoding) as f:
        config = yaml.safe_load(f)
    return config