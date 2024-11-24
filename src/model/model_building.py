import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml

#train_data  = pd.read_csv("./data/processed/train_processed_mean.csv") # C:\Users\honor\Desktop\ml_pipeline_papka_na_C_PC\data\processed\train_processed_mean.csv
def load_data(filepath):
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath} : {e}")


##X_train = train_data.iloc[:,0:-1].values # берем все строки и все столбцы (кроме столбца target, y_train) из train_data
##y_train = train_data.iloc[:,-1].values   # наш целевой (target, y_train) столбец из train_data
## Для корректной работы API мы видлизменяем наш формат на DataFrame pandas:
#X_train = train_data.drop(columns = ['Potability'], axis = 1)
#y_train = train_data['Potability']
def prepare_data(data): # data = train_data
    try:
        X_train = data.drop(columns = ['Potability'], axis = 1)
        y_train = data['Potability']
        return X_train, y_train
    except Exception as e:
        raise Exception(f"Error Preparing data : {e}")

#n_estimators = yaml.safe_load(open("params.yaml", "r"))["model_building"]["n_estimators"]
def load_params(params_path):
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        return params["model_building"]["n_estimators"]
    except Exception as e:
        raise Exception(f"Error loading parameters from {params_path} : {e}")

#clf = RandomForestClassifier(n_estimators=n_estimators)
#clf.fit(X_train, y_train)
def train_model(X_train, y_train, n_estimators):
    try:
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X_train, y_train)
        return clf
    except Exception as e:
        raise Exception(f"Error Trainig model : {e}")



# Теперь сохраним нашу модель с помощью pickle:
#pickle.dump(clf, open("model.pkl", "wb"))
def save_model(clf_model, model_name):
    try:
        with open(model_name, "wb") as file:
            pickle.dump(clf_model, file)
    except Exception as e:
        raise Exception(f"Error saving model to {model_name} : {e}")

##############
def main():
    data_filepath = "./data/processed/train_processed_median.csv"
    params_filepath = "params.yaml"
    model_name = "models/model.pkl" # тут вместо "model.pkl" пишем уже "models/"model.pkl"
    try:
        train_data = load_data(data_filepath)
        n_estimators = load_params(params_filepath) # 100
        X_train, y_train = prepare_data(train_data)
        clf_model = train_model(X_train, y_train, n_estimators)
        save_model(clf_model, model_name)
    except Exception as e:
        raise Exception(f"An error occurred : {e}")
if __name__ == "__main__":
    main()