from fastapi import FastAPI, File, Form, UploadFile
from io import StringIO
import requests
#import uvicorn
#from pydantic import BaseModel
import pickle
import pandas as pd
from data_model import Water
#import requests
# Создаем экземпляр с именем FastAPI,
# и прописываем название этого API (title) и его описание (description)
app = FastAPI(title = "Water Potability Prediction",
              description = "Predicting Water Potability")

# Теперь загрузим предварительно обученную модель - а именно это файл model.pkl
# Сначала нам нудно открыть этот файл model.pkl, поэтому:
with open(r"C:\Users\honor\dvc_github_repo\wpotab_github24\models\model.pkl", "rb") as f:
    model = pickle.load(f) # загружем непосредственно эту модель

# Теперь мы создадим конечную точку, которая будем выступать в качестве Домашней Страницы нашего API:
@app.get("/") # если теперь кто-то перейдем по этой конечной точке, то будем вызвана следующая функция:
def index():
    return "Welcome to Water Potability Prediction FastAPI" # и мы вернем приветственное сообщение
# Вот мы создали нашу первую конечную точку, которая будем выступать в качестве Домашней Страницы нашего API.

# Теперь мы создадим Вторую Конечную Точку (Первая Конечная Точка у нас уже есть - и это наша Стартовая (Домашняя Станица).
# Вторая же Конечная Точка будет отвечать за предсказание, т.е. сюда вот мы будем подавать наши данные:
@app.post("/predict")
def model_predict(water : Water): # мы можем тут в скобках передавать наши данные, но это усложнит код. Поэтому мы сделаем иначе:
    sample = pd.DataFrame({
        'ph' : [water.ph],
        'Hardness' : [water.Hardness],
        'Solids' : [water.Solids],
        'Chloramines' : [water.Chloramines],
        'Sulfate' : [water.Sulfate],
        'Conductivity' : [water.Conductivity],
        'Organic_carbon' : [water.Organic_carbon],
        'Trihalomethanes' : [water.Trihalomethanes],
        'Turbidity' : [water.Turbidity]
    }) 
    
    predicted_value = model.predict(sample)
    # Если в результате предсказания у нас для какого-то забора воды (клиента елси кредитный риск) выпала 1, то
    # # вода пригодна для употребления,
    # # иначе, если выпало НЕ 1, вода не пригодня для употребления:
    if predicted_value == 1:
        return "Water is Consumable_Voda_Prigodna_dlya_ypotreblenia"
    else:
        return "Water is not Consumable_Voda_Ne_Prigodna_dlya_ypotreblenia"

#@app.post("/files/")
#async def batch_prediction(file: bytes = File(...)):
#    s=str(file,'utf-8')
#    data = StringIO(s)
#    df=pd.read_csv(data)
#    lst = df.values.tolist()
#    inference_request = {
#        "dataframe_records": lst
#    }
#    endpoint = 'http://localhost:1234/invocations'
#    response = requests.post(endpoint, json = inference_request)
#    print(response.text)
#    return response.text

