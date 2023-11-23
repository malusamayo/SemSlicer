import json
import pandas as pd

def read_txt_file(path):
    with open(path, 'r') as f:
        data = f.readlines()
    data = [item.strip() for item in data]
    return data

def read_csv_file(path):
    df = pd.read_csv(path)
    return df