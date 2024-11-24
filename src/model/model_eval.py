import numpy as np
import pandas as pd

import pickle
import json
################### Для DVCLive начало:
from dvclive import Live
import yaml
################### Для DVCLive конец.

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
#test_data = pd.read_csv("./data/processed/test_processed_mean.csv")
def load_data(test_processed_filepath):
    try:
         return pd.read_csv(test_processed_filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {test_processed_filepath}:{e}")

# Берем то же самое что бы ранее писали в файле model_building.py, но меняем train на test уже
#X_test = test_data.iloc[:,0:-1].values # берем все строки и все столбцы (кроме столбца target, y_test) из test_data
#y_test = test_data.iloc[:,-1].values   # наш целевой (target, y_test) столбец из test_data

def prepare_data(Test_data):
    try:
        X_test = Test_data.drop(columns=['Potability'],axis=1)
        y_test = Test_data['Potability']
        return X_test, y_test
    except Exception as e:
        raise Exception(f"Error Preparing data:{e}")

# Теперь заружаем ранее созданную в файле model_building.py модель, которуя представляет собойт отдельный файл
# model.pkl, который у нас повился после dvc repro на этапе model_building
#model = pickle.load(open("model.pkl", "rb"))
def load_model(model_name_or_model_filepath):
    try:
        with open(model_name_or_model_filepath,"rb") as file:
            model= pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {model_name_or_model_filepath}:{e}")

# Теперь сделаем прогнозирование на наших данных X_test
#y_pred = model.predict(X_test)
# И теперь найдем значение показателя Точности (Accuracy Score)
#acc = accuracy_score(y_test, y_pred) # y_test = наши фактические данные, y_pred = наши прогнозные данные, полученные на основе модели
#pre = precision_score(y_test, y_pred)
#recall = recall_score(y_test, y_pred)
#f1score = f1_score(y_test, y_pred)
# Теперь сохраним эти данные (по acc, pre, recall, f1score) в формте JSON,
# для этого создадим словарь metrics_dict
#metrics_dict = {
#    'acc':acc,
#    'precision':pre,
#    'recall' : recall,
#    'f1_score': f1score}

def evaluation_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        ################### Для DVCLive начало:
        params = yaml.safe_load(open("params.yaml", "r"))
        test_size = params["data_collection"]["test_size"]
        n_estimators = params["model_building"]["n_estimators"]
        ################### Для DVCLive конец.
        acc = accuracy_score(y_test,y_pred)
        pre = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1score = f1_score(y_test,y_pred)
        ################### Для DVCLive начало:
        with Live(save_dvc_exp = True) as live:
            live.log_metric("acc", acc)
            live.log_metric("pre", pre)
            live.log_metric("recall", recall)
            live.log_metric("f1score", f1score)
            live.log_param("test_size", test_size)
            live.log_param("n_estimators", n_estimators)
        ################### Для DVCLive конец.

        metrics_dict = {

            'acc':acc,
            'precision':pre,
            'recall' : recall,
            'f1_score': f1score
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error evaluating model : {e}")


# Теперь создадим непосредственно сам файл JSON, и запишем туда наши данные по метрикам:
#with open("metrics.json", "w") as file:
#    json.dump(metrics_dict, file, indent=4)
# Мы создали данный фалй с метриками в формате JSON, чтобы мы могли после всех наших проделанных этапов
# написать в VSCODE команду dvc metrics show, и увидеть значения по всем метрикам для данной модели.
def save_metrics(metrics, metrics_path):
    try:
        with open(metrics_path,'w') as file:
            json.dump(metrics,file,indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {metrics_path}:{e}")
    
##############
def main():
    try:
        test_processed_filepath = "./data/processed/test_processed_mean.csv"
        model_name_or_model_filepath = "models/model.pkl" # тут вместо "model.pkl" пишем уже "models/model.pkl"
        metrics_path = "reports/metrics.json" # тут вместо "metrics.json" пишем уже "reports/metrics.json"

        Test_data = load_data(test_processed_filepath)
        X_test,y_test = prepare_data(Test_data)
        model = load_model(model_name_or_model_filepath)
        metrics = evaluation_model(model,X_test,y_test)
        save_metrics(metrics,metrics_path)
    except Exception as e:
        raise Exception(f"An Error occurred:{e}")

if __name__ == "__main__":
    main()