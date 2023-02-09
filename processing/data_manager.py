import pandas as pd
import pickle
import os
import yaml
from pathlib import Path
import json

# from src.logger import Logger

# logger = Logger('utilities_logs')

def get_project_root(root_name = 'product_recommender') -> Path:
    dir = Path(__file__)
    while True:
        if root_name not in dir.name.lower() :
            dir = dir.parent
        else:
            return dir

def load_yaml_config():
    yaml_config = None    
    #if not os.path.exists("config/config.yaml"):
    #    Path("config/config.yaml").touch()
    configuration_path = get_project_root().joinpath("config/config.yaml")
    try:
        with open(configuration_path, "r", encoding="utf8") as file:
            try:
                yaml_config = yaml.load(file, Loader=yaml.FullLoader)
                return yaml_config
            except yaml.YAMLError as error:
                raise error
    except FileNotFoundError as error:
        raise error

def get_config() -> object:
    config = load_yaml_config()
    return config

def read_files(filename):
    csv_file = pd.read_csv(filename) 
    return csv_file

def save_files(data,save_path):
    data.to_csv(save_path,index=False)

def save_model(model,save_path):
    #pickle.dumps(model,save_path)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)

def load_model(file_name):
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model     
    #pickle.load(file_name=file_name)