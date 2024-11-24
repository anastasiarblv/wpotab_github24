from pydantic import BaseModel
class Water(BaseModel): # Данные Класс определяет названия столбцов и их тип
    # Прописываем какой тип у каждого столбца наших данных:
    ph : float
    Hardness : float
    Solids : float
    Chloramines : float
    Sulfate : float
    Conductivity : float
    Organic_carbon : float
    Trihalomethanes : float
    Turbidity : float
