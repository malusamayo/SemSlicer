import json

def read_txt_file(path):
    with open(path, 'r') as f:
        data = f.readlines()
    data = [item.strip() for item in data]
    print("read {} lines from {}".format(len(data), path))
    return data
