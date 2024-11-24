import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml
  
#data = pd.read_csv(r"C:\Users\honor\Desktop\water_potability.csv")
def load_data(filepath):
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")

#test_size = yaml.safe_load(open("params.yaml"))["data_collection"]["test_size"]
def load_params(filepath): # путь к исходному входном файлу
    try:
        with open(filepath, "r") as file: # открываем данный файл в режиме чтения "r"
            params = yaml.safe_load(file)
        return params["data_collection"]["test_size"]
    except Exception as e:
        raise Exception(f"Error loading parameters from {filepath}: {e}")


# train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
def split_data(data, test_size):
    try:
        return train_test_split(data, test_size=test_size, random_state=42)
    except ValueError as e:
        raise ValueError(f"Error splitting data : {e}")



#data_path = os.path.join("data", "raw")
#os.makedirs(data_path)
#train_data.to_csv(os.path.join(data_path, "train.csv"), index = False)
#test_data.to_csv(os.path.join(data_path, "test.csv"), index = False)
def save_data(df, filepath):
    try:
        df.to_csv(filepath, index = False)
    except Exception as e:
        raise Exception(f"Error saving parameters to {filepath}: {e}")

##############
def main():
    data_filepath = r"C:\Users\honor\Desktop\water_potability.csv"
    params_filepath = "params.yaml"
    raw_data_path = os.path.join("data", "raw") # путь, куда мы помещаем наши обработанные данные после этого этапа (стадии)
    # Внутри этой функции main мы запускаем все наши функции
    try:
        data = load_data(data_filepath)
        test_size = load_params(params_filepath) # 0.20
        train_data, test_data = split_data(data, test_size)
        os.makedirs(raw_data_path)
        save_data(train_data, os.path.join(raw_data_path, "train.csv"))
        save_data(test_data, os.path.join(raw_data_path, "test.csv"))
    except Exception as e:
        raise Exception(f"An error occurred : {e}")
if __name__ == "__main__":
    main()